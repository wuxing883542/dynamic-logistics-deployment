import sys
import os
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig


class RobustHubEnv(gym.Env):
    """
    半自回归动态调度 PPO 环境 —— 需求时钟 + 完备 MDP (时序公平优化版 + 防梯度爆炸)
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

        self.coords = self.topo['coords']
        self.dist_matrix = self.topo['C']
        self.node_types = self.topo['node_types']
        self.base_intensity = self.topo['base_intensity']

        # ── 2. 静态枢纽与空间掩码 ──
        self.hub_locations = self._generate_fixed_hubs()
        self.max_radius = getattr(self.cfg, 'max_flight_radius', 750.0)
        self.action_mask = self._build_spatial_action_mask()

        # ── 3. 状态变量 ──
        self.current_scenario = None
        self.current_t = 0
        self.hub_capacities = np.zeros(self.K)
        self.ep_total_demand = 0.0
        self.ep_total_unmet = 0.0
        self.day_total_expected_demand = 0.0
        self.current_predicted_demand = 0.0

        # ── 4. 动作与观测空间 ──
        self.action_space = spaces.MultiDiscrete([self.K + 1] * self.N)

        self.observation_space = spaces.Dict({
            'node_features':   spaces.Box(low=-1.0, high=1.0, shape=(self.N, 6), dtype=np.float32),
            'current_orders':  spaces.Box(low=0.0, high=np.inf, shape=(self.N,), dtype=np.float32),
            'hub_mask':        spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            'hub_capacities':  spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32),
            'time_ratio':      spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'macro_pressure':  spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            'future_pressure': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            'static_target':   spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), # [执行方案B]
        })

    # ── 枢纽生成 ──────────────────────────────────────────────
    def _generate_fixed_hubs(self):
        kmeans = KMeans(n_clusters=self.K, random_state=self.cfg.seed, n_init=10)
        labels = kmeans.fit_predict(self.coords)

        fixed_hubs = []
        for k in range(self.K):
            cluster_indices = np.where(labels == k)[0]
            best_local_idx = np.argmax(self.base_intensity[cluster_indices])
            best_global_idx = cluster_indices[best_local_idx]
            fixed_hubs.append(int(best_global_idx))

        print("=" * 60)
        print(f"[环境初始化 ({self.mode})] K-Means 划分 {self.K} 个片区，静态枢纽锁定:")
        for i, hub_id in enumerate(fixed_hubs):
            ntype = "商业区 UMa" if self.node_types[hub_id] == 1 else "住宅区 UMi"
            print(f"    枢纽 {i}: 节点 [{hub_id:03d}] | 类型: {ntype} | 强度: {self.base_intensity[hub_id]:.2f}")
        print("=" * 60)
        return fixed_hubs

    # ── 动作掩码 (升级版：同地订单强制绑定) ──
    def _build_spatial_action_mask(self):
        mask = np.ones((self.N, self.K + 1), dtype=bool)
        for i in range(self.N):
            is_hub_itself = False
            my_hub_idx = -1
            for k in range(self.K):
                if i == self.hub_locations[k]:
                    is_hub_itself = True
                    my_hub_idx = k
                    break

            if is_hub_itself:
                # 💡 如果我自己就是枢纽，只能派给自己，连拒单也不行
                for k in range(self.K):
                    if k != my_hub_idx:
                        mask[i, k] = False
                mask[i, self.K] = False 
            else:
                # 💡 普通节点，走距离判定逻辑
                for k in range(self.K):
                    if self.dist_matrix[i, self.hub_locations[k]] > self.max_radius:
                        mask[i, k] = False
                mask[i, self.K] = True 

        return mask

    def get_action_mask(self):
        return self.action_mask.copy()

    # ── 预测信息接口 ────────────────────────────────────────────
    def set_predictor_info(self, predicted_total_demand):
        self.current_predicted_demand = predicted_total_demand

    # ── 环境重置 ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        scenarios = self.train_scenarios if self.mode == 'train' else self.eval_scenarios
        day_idx = self.np_random.integers(len(scenarios))
        self.current_scenario = scenarios[day_idx]

        self.day_total_expected_demand = float(np.sum(self.current_scenario))
        self.current_t = 0
        self.hub_capacities = np.full(self.K, self.cfg.Q, dtype=np.float32)
        self.ep_total_demand = 0.0
        self.ep_total_unmet = 0.0
        self.current_predicted_demand = 0.0

        # 💡 [新增] 计算全天静态公平基准线（基于初始满载运力）
        total_initial_cap = self.K * self.cfg.Q
        self.day_static_target = min(1.0, total_initial_cap / max(self.day_total_expected_demand, 1.0))

        return self._get_obs(), {}

    # ── 观测构建 ──────────────────────────────────────────────
    def _get_obs(self):
        node_f = np.zeros((self.N, 6), dtype=np.float32)
        node_f[:, 0] = self.coords[:, 0] / self.cfg.map_size
        node_f[:, 1] = self.coords[:, 1] / self.cfg.map_size
        node_f[:, 2] = self.node_types.astype(np.float32)
        max_intensity = max(np.max(self.base_intensity), 1.0)
        node_f[:, 3] = self.base_intensity / max_intensity
        center = np.array([self.cfg.map_size * 0.5, self.cfg.map_size * 0.5])
        node_f[:, 4] = np.linalg.norm(self.coords - center, axis=1) / (self.cfg.map_size * 0.707)

        orders = (self.current_scenario[self.current_t].astype(np.float32)
                  if self.current_t < self.T
                  else np.zeros(self.N, dtype=np.float32))

        active_idx = np.where(orders > 0)[0]
        if len(active_idx) > 0:
            ranks = np.argsort(np.argsort(orders[active_idx]))
            norm_ranks = ranks / max(1, len(active_idx) - 1)
            node_f[active_idx, 5] = norm_ranks

        hub_mask = np.isin(np.arange(self.N), self.hub_locations).astype(np.float32)

        total_rem_cap = np.sum(self.hub_capacities)
        consumed_cap_ratio = 1.0 - (total_rem_cap / max(self.K * self.cfg.Q, 1.0))
        demand_arrival_ratio = self.ep_total_demand / max(self.day_total_expected_demand, 1.0)
        macro_pressure = max(0.0, consumed_cap_ratio - demand_arrival_ratio)
        # 💡 2. 新增：前瞻预测压力（独立信号）
        # ⚠️ 必须加上 max(total_rem_cap, 1.0) 防止晚高峰运力耗尽时除以 0 导致 nan！
        future_pressure = float(self.current_predicted_demand) / max(total_rem_cap, 1.0)
        # 稍微做个截断，保护神经网络不被极端值击穿
        future_pressure = min(future_pressure, 5.0)


        return {
            'node_features':   node_f,
            'current_orders':  orders,
            'hub_mask':        hub_mask,
            'hub_capacities':  self.hub_capacities.copy() / max(self.cfg.Q, 1.0),
            'time_ratio':      np.array([self.current_t / self.T], dtype=np.float32),
            'macro_pressure':  np.array([macro_pressure], dtype=np.float32),
            'future_pressure': np.array([future_pressure], dtype=np.float32), # 💡 并列送入
            'static_target':   np.array([getattr(self, 'day_static_target', 1.0)], dtype=np.float32), # [执行方案B]
        }

    # ── 单步推演 ──────────────────────────────────────────────
    # 原本是：def step(self, action):
    # 改为（默认 True，保证训练和其他地方完全不变）：
    def step(self, action, enable_local_free=True):
        action = np.asarray(action, dtype=np.int32).flatten()
        orders = self.current_scenario[self.current_t]
        active_nodes = np.where(orders > 0)[0]
        step_demand = float(np.sum(orders))

        # ── 1. 预计算需求时钟与“时序公平目标” (CE-MPC) ──
        pre_action_rem_cap = float(np.sum(self.hub_capacities))
        
        # 预估当前到午夜的剩余总需求 (利用先验分布/预测网络)
        rem_expected_demand = float(self.day_total_expected_demand - self.ep_total_demand)
        
        # ── 1. 获取全局静态时序公平目标 ──
        # 💡 [修改] 废弃可被智能体操控的动态 target，使用全天死任务
        target_cov = self.day_static_target

        # ── 2. 执行分配 (双轨制：无人机 vs 电梯免单) ──
        step_allocated = 0.0
        allocated_per_node = np.zeros(self.N, dtype=np.float32)

        if len(active_nodes) > 0:
            hub_requests = {k: 0.0 for k in range(self.K)}
            
            # 第一轮：只统计异地订单对无人机的运力请求
            for i in active_nodes:
                hub_choice = action[i]
                if hub_choice != self.K:  
                    dist = self.dist_matrix[i, self.hub_locations[hub_choice]]
                    if dist <= self.max_radius:
                        if i != self.hub_locations[hub_choice]: 
                            hub_requests[hub_choice] += orders[i]

            # 计算无人机运力挤兑比例
            hub_alloc_ratio = np.zeros(self.K, dtype=np.float32)
            for k in range(self.K):
                if hub_requests[k] > 0:
                    if hub_requests[k] <= self.hub_capacities[k]:
                        hub_alloc_ratio[k] = 1.0 
                    else:
                        hub_alloc_ratio[k] = self.hub_capacities[k] / hub_requests[k] 

            # 第二轮：实际结算
            for i in active_nodes:
                hub_choice = action[i]
                if hub_choice != self.K:
                    dist = self.dist_matrix[i, self.hub_locations[hub_choice]]
                    if dist <= self.max_radius:
                        if i == self.hub_locations[hub_choice] and enable_local_free:
                            # 【同地免单】
                            alloc_vol = float(orders[i])
                            allocated_per_node[i] = alloc_vol
                            step_allocated += alloc_vol
                        else:
                            # 【异地派送】扣除运力
                            alloc_vol = orders[i] * hub_alloc_ratio[hub_choice]
                            allocated_per_node[i] = alloc_vol
                            self.hub_capacities[hub_choice] -= alloc_vol
                            step_allocated += alloc_vol
        # ── 3. 更新累计统计 ──
        self.ep_total_demand += step_demand
        self.ep_total_unmet += (step_demand - step_allocated)

        # 💡 [修复假摔Bug] 如果该步需求为 0，覆盖率直接视为 100% (1.0)
        if step_demand <= 0:
            step_cov = 1.0
        else:
            step_cov = step_allocated / step_demand

        # ── 4. 目标追踪混合奖励 (Target-Tracking Hybrid Reward) ──
        node_rewards = np.zeros(self.N, dtype=np.float32)

        # 【恢复原始逻辑】抛弃多余的时间衰减，信任 target_cov 自身的闭环调节能力
        base_c_fair = 10.0
        # 💡 [二阶优化：单边惩罚] 
        # 只惩罚没达标的；如果因为同地免单导致超额完成，不仅不罚，反而给一点微弱奖励
        if step_cov < target_cov:
            fairness_error = target_cov - step_cov
            global_fairness_reward = -(fairness_error ** 2) * base_c_fair
        else:
            # 超标了（覆盖率高于及格线），给一点正向激励，系数 2.0 可以根据需要微调
            global_fairness_reward = (step_cov - target_cov) * 2.0


        for i in active_nodes:
            hub_choice = action[i]
            
            # 微观项 1：空间路由效率。在公平分配的前提下，鼓励分配给距离最近的枢纽。
            local_routing_reward = 0.0
            if hub_choice != self.K and allocated_per_node[i] > 0:
                dist = self.dist_matrix[i, self.hub_locations[hub_choice]]
                dist_score = 1.0 - (dist / self.max_radius)
                local_routing_reward = dist_score * 0.5  # 最大奖励 0.5
                
            # 微观项 2：体量阻尼。用于在网络决定“该拒掉谁”来满足 target_cov 时，优先保留大体量节点。
            missed = orders[i] - allocated_per_node[i]
            volume_penalty = -(missed / max(step_demand, 1e-5)) * 1.0

            # CTDE (集中式训练分布式执行) 终极组装
            # 权重保证了：维护时序公平 (全局) > 空间距离效率 (微观) > 单点损失 (防大节点被秒)
            node_rewards[i] = global_fairness_reward + local_routing_reward + volume_penalty

        # ── 5. 时钟推进 ──
        self.current_t += 1
        terminated = self.current_t >= self.T
        ep_cov = (1.0 - (self.ep_total_unmet / max(self.ep_total_demand, 1e-5))
                  if terminated else 0.0)

        # 💡 [二阶优化：期末运力清算]
        terminal_waste_penalty = 0.0
        if terminated:
            total_rem_cap = np.sum(self.hub_capacities)
            total_initial_cap = self.K * self.cfg.Q
            waste_ratio = total_rem_cap / max(total_initial_cap, 1.0)
            # 如果全天结束运力没花完，给予终极重罚
            terminal_waste_penalty = -waste_ratio * 50.0 
            
            # 把惩罚均摊给所有节点
            if len(active_nodes) > 0:
                node_rewards[active_nodes] += terminal_waste_penalty / len(active_nodes)



        info = {
            't':              self.current_t,
            'unmet_penalty':  (step_demand - step_allocated) * self.cfg.penalty_unmet,
            'step_demand':    step_demand,
            'step_unmet':     step_demand - step_allocated,
            'step_coverage':  step_cov,
            'ep_coverage':    ep_cov,
            'node_rewards':   node_rewards,
        }

        return self._get_obs(), float(np.mean(node_rewards)), terminated, False, info

if __name__ == '__main__':
    print(">>> 正在启动环境实例化测试...")
    cfg = UAVHubConfig()
    env = RobustHubEnv(cfg, mode='train')
    obs, _ = env.reset()
    print(">>> 环境实例化与 Reset 成功，时序公平需求时钟(防爆版)已激活！")