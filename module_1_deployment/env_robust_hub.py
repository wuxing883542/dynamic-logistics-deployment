import sys
import os
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
from sklearn.cluster import KMeans

# 确保能找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig


class RobustHubEnv(gym.Env):
    """
    单阶段动态调度 PPO 环境（纯粹分配）
    
    核心特性：
    1. 彻底解耦选址：初始化时利用 K-Means + 峰值贪婪锁定 K 个枢纽。
    2. 真·马尔可夫决策：运力在全天真实累积消耗，迫使 AI 学会“囤积运力”。
    3. IPPO (多智能体信用分配)：每个节点独立核算奖励，解决联合动作空间下的吃大锅饭问题！
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: UAVHubConfig, mode: str = 'train'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        # ── 1. 加载数据 ──
        data_path = os.path.join(BASE_DIR, 'data', f'map_adaptive_seed{cfg.seed}_robust.pkl')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.topo = data['topo_data']
        self.train_scenarios = data['train_scenarios']   
        self.eval_scenarios = data.get('eval_scenarios', data['train_scenarios'][-30:])
        self.N = data['config_meta']['N']
        self.T = cfg.T_timesteps
        self.K = cfg.max_hubs

        # 提取关键节点特征
        self.coords = self.topo['coords']
        self.dist_matrix = self.topo['C']
        self.node_types = self.topo['node_types']
        self.base_intensity = self.topo['base_intensity']

        # ── 2. 自动生成静态枢纽与物理掩码 ──
        self.hub_locations = self._generate_fixed_hubs()
        self.max_radius = getattr(self.cfg, 'max_flight_radius', 750.0)
        self.action_mask = self._build_spatial_action_mask()

        # ── 3. 状态与业务统计变量 ──
        self.current_scenario = None   
        self.current_t = 0             
        self.hub_capacities = np.zeros(self.K)
        
        self.ep_total_demand = 0.0
        self.ep_total_unmet = 0.0

        # ── 4. 动作与观测空间设计 ──
        self.action_space = spaces.MultiDiscrete([self.K + 1] * self.N)

        self.observation_space = spaces.Dict({
            'node_features':   spaces.Box(low=-1.0, high=1.0, shape=(self.N, 5), dtype=np.float32),
            'current_orders':  spaces.Box(low=0.0, high=np.inf, shape=(self.N,), dtype=np.float32),
            'hub_mask':        spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            'hub_capacities':  spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32),
        })

    def _generate_fixed_hubs(self):
        kmeans = KMeans(n_clusters=self.K, random_state=self.cfg.seed, n_init=10)
        labels = kmeans.fit_predict(self.coords)

        fixed_hubs = []
        for k in range(self.K):
            cluster_indices = np.where(labels == k)[0]
            best_local_idx = np.argmax(self.base_intensity[cluster_indices])
            best_global_idx = cluster_indices[best_local_idx]
            fixed_hubs.append(int(best_global_idx))
            
        print("="*60)
        print(f"🎯 [环境初始化 ({self.mode})] 自动划分 {self.K} 个片区，静态枢纽已锁定：")
        for i, hub_id in enumerate(fixed_hubs):
            node_type = "商业区 UMa" if self.node_types[hub_id] == 1 else "住宅区 UMi"
            print(f"   ➤ 枢纽 {i}: 节点 [{hub_id:03d}] | 类型:{node_type} | 强度:{self.base_intensity[hub_id]:.2f}")
        print("="*60)
        return fixed_hubs

    def _build_spatial_action_mask(self):
        mask = np.ones((self.N, self.K + 1), dtype=bool)
        for i in range(self.N):
            for k in range(self.K):
                if self.dist_matrix[i, self.hub_locations[k]] > self.max_radius:
                    mask[i, k] = False
            mask[i, self.K] = True 
        return mask

    def get_action_mask(self):
        return self.action_mask.copy()

    # ── 环境重置 ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        scenarios = self.train_scenarios if self.mode == 'train' else self.eval_scenarios
        day_idx = self.np_random.integers(len(scenarios))
        self.current_scenario = scenarios[day_idx]

        self.current_t = 0
        self.hub_capacities = np.full(self.K, self.cfg.Q)
        self.ep_total_demand = 0.0
        self.ep_total_unmet = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        # 💡 [改动 1] 维度从 5 改成 6
        node_f = np.zeros((self.N, 6), dtype=np.float32)
        node_f[:, 0] = self.coords[:, 0] / self.cfg.map_size          
        node_f[:, 1] = self.coords[:, 1] / self.cfg.map_size          
        node_f[:, 2] = self.node_types.astype(np.float32)             
        max_intensity = max(np.max(self.base_intensity), 1.0)
        node_f[:, 3] = self.base_intensity / max_intensity
        center = np.array([self.cfg.map_size * 0.5, self.cfg.map_size * 0.5])
        node_f[:, 4] = np.linalg.norm(self.coords - center, axis=1) / (self.cfg.map_size * 0.707)

        orders = self.current_scenario[self.current_t].astype(np.float32) if self.current_t < self.T else np.zeros(self.N, dtype=np.float32)

        # 💡 [改动 2] 新增第 6 维：当前时刻的“相对抢单优先级” (0.0 到 1.0)
        # 需求越大的节点，得分越接近 1.0，让网络知道“我会被环境优先处理，我能抢到枢纽！”
        active_idx = np.where(orders > 0)[0]
        if len(active_idx) > 0:
            # 两次 argsort 可以得到每个元素的排名
            ranks = np.argsort(np.argsort(orders[active_idx])) 
            norm_ranks = ranks / max(1, len(active_idx) - 1)
            node_f[active_idx, 5] = norm_ranks

        hub_mask = np.zeros(self.N, dtype=np.float32)
        for h in self.hub_locations:
            hub_mask[h] = 1.0

        return {
            'node_features':  node_f,
            'current_orders': orders,
            'hub_mask':       hub_mask,
            'hub_capacities': self.hub_capacities.copy() / max(self.cfg.Q, 1.0),
        }
    # ── 物理推演与结算 ───────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.int32).flatten()
        orders = self.current_scenario[self.current_t]
        active_nodes = np.where(orders > 0)[0]

        transport_cost = 0.0
        unmet_penalty = 0.0
        
        step_demand = float(np.sum(orders))
        step_allocated = 0.0
        step_unmet = 0.0

        # 💡 [IPPO核心修改] 为每个节点设立独立的“账本”
        per_node_cost = np.zeros(self.N, dtype=np.float32)

        sorted_idx = np.argsort(-orders[active_nodes])

        for idx in sorted_idx:
            i = active_nodes[idx]
            demand = orders[i]
            hub_choice = action[i]

            if hub_choice == self.K:
                step_unmet += demand
                penalty = demand * self.cfg.penalty_unmet
                unmet_penalty += penalty
                per_node_cost[i] += penalty  # 记入独立账本
            else:
                dist = self.dist_matrix[i, self.hub_locations[hub_choice]]
                
                if dist > self.max_radius:
                    allocated = 0.0
                else:
                    allocated = min(demand, self.hub_capacities[hub_choice])

                cost = allocated * dist * 0.01
                transport_cost += cost
                per_node_cost[i] += cost     # 记入独立账本
                self.hub_capacities[hub_choice] -= allocated  
                step_allocated += allocated
                
                if allocated < demand:
                    missed = demand - allocated
                    step_unmet += missed
                    penalty = missed * self.cfg.penalty_unmet
                    unmet_penalty += penalty
                    per_node_cost[i] += penalty # 记入独立账本

        self.ep_total_demand += step_demand
        self.ep_total_unmet += step_unmet

        # =======================================================
        # 👑 多智能体独立奖励 (Per-Node Reward) - IPPO灵魂
        # =======================================================
        
        # 1. 基础惩罚：每个节点只为自己的运费和拒单买单 (完美切断大锅饭)
        # 量级对齐百万级缩放
        node_rewards = -per_node_cost / 1000000.0
        
        # 2. 全局分红：大盘利用率当成系统奖金，平分给所有人
        utilization_reward = 0.01 * (step_allocated / max(step_demand, 1.0))
        node_rewards += utilization_reward

        self.current_t += 1
        terminated = self.current_t >= self.T

        # 3. 终局大奖也是阳光普照
        if terminated:
            final_coverage = 1.0 - (self.ep_total_unmet / max(self.ep_total_demand, 1e-5))
            jackpot_reward = 5.0 * (final_coverage ** 10)
            node_rewards += jackpot_reward
        else:
            final_coverage = 0.0

        info = {
            't':               self.current_t,
            'transport_cost':  transport_cost,
            'unmet_penalty':   unmet_penalty,
            'step_cost':       transport_cost + unmet_penalty,
            'step_demand':     step_demand, 
            'step_unmet':      step_unmet,  
            'ep_coverage':     final_coverage if terminated else 0.0,
            'node_rewards':    node_rewards # 💡 传出独立奖励数组
        }

        # gym 要求 step 返回标量 reward，返回均值应付 API (训练不用它)
        return self._get_obs(), float(np.mean(node_rewards)), terminated, False, info

if __name__ == '__main__':
    print(">>> 正在启动环境实例化测试...")
    cfg = UAVHubConfig()
    env = RobustHubEnv(cfg, mode='train')
    obs, _ = env.reset()
    print(">>> 环境实例化与 Reset 成功，全局时空运筹与 IPPO 模式已激活！")