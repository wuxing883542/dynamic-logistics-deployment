import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os

# 确保导入路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig
from utils.physics_model import UAVPhysicsModel

class RobustHubEnv(gym.Env):
    """
    面向潮汐需求的无人机枢纽鲁棒选址环境 (V9.0 究极数学自洽版)
    核心特性：
    1. 物理对齐：全面接入自定义 physics_model 计算严谨的星型直飞经济成本。
    2. 梯度平滑：训练期使用 Sinkhorn 软分配期望值进行平滑结算。
    3. 科学基准：采用随机策略采样作为动态分母，彻底解决通货膨胀。
    4. 闲置惩罚：精细到逐个枢纽的微观运力闲置税，拒绝一刀切。
    5. 完美覆盖率：基于“大道至简”的物理守恒定律，彻底根治覆盖率溢出。
    6. 鲁棒硬约束：引入成本标准差惩罚，并在 95% 全覆盖红线下寻找最优解。
    """
    def __init__(self, cfg: UAVHubConfig):
        super(RobustHubEnv, self).__init__()
        self.cfg = cfg
        # 实例化用户自定义的物理模型引擎
        self.physics = UAVPhysicsModel(cfg)
        
        # 1. 加载地形与潮汐考卷
        data_path = os.path.join(BASE_DIR, 'data', f'map_{cfg.N}n_seed{cfg.seed}_robust.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.topo = data['topo_data']
        self.snapshots = data['snapshots_total'] 
        self.N = cfg.N
        self.coords = self.topo['coords']
        self.dist_matrix = self.topo['C'] 
        self.fixed_costs = self.topo['f'] 

        # ==================================================
        # 🎯 物理规则与奖惩系数动态设定
        # ==================================================
        # Sinkhorn 温度参数：控制概率分布的敏锐度
        self.sinkhorn_temp = 5.0
        
        # 影子价格更新步长：控制对拥堵(超载)反应的剧烈程度
        self.shadow_step = 0.1 
        
        # 🚨 超载红线机制：惩罚设为违约金的 2 倍
        self.overload_coef = self.cfg.penalty_unmet * 2.0 

        # ==================================================
        # 🎯 史上最科学的成本基准 (Baseline Cost)
        # ==================================================
        print("⏳ 正在测算环境的科学成本基准 (Baseline Sampling)...")
        baseline_costs = []
        self.lambda_robust = 1.5  # 💡 提前定义鲁棒权重，确保全局统一
        
        for _ in range(5):
            # 随机瞎选 max_hubs 个站
            random_hubs = np.random.choice(self.N, self.cfg.max_hubs, replace=False)
            random_fixed_cost = sum([self.fixed_costs[h] for h in random_hubs])
            
            # 💡 [补充修复]：基准测算也使用全局抽样，避免被早高峰带偏
            sample_indices = np.random.choice(len(self.snapshots), 20, replace=False)
            target_snaps_baseline = [self.snapshots[i] for i in sample_indices]
            
            # 💡 [修复]：基准测算也必须同步计算波动率 (std)
            snap_costs_for_baseline = []
            
            for snap in target_snaps_baseline:
                # 注意：_sinkhorn_allocation 现在返回 (纯运营成本, 覆盖率, 闲置税)
                op_cost, _, idle_p = self._sinkhorn_allocation(random_hubs, snap)
                snap_costs_for_baseline.append(op_cost + idle_p)
                
            avg_op_and_idle = np.mean(snap_costs_for_baseline)
            std_cost = np.std(snap_costs_for_baseline)
            
            # 分母彻底对齐分子：建站费 + 平均运营 + λ*波动率
            baseline_costs.append(random_fixed_cost + avg_op_and_idle + (self.lambda_robust * std_cost))
                
        self.baseline_cost = np.mean(baseline_costs)
        print(f"✅ 基准成本测算完成: {self.baseline_cost:.2f} (将作为奖励归一化分母)")

        # ==================================================
        # 🎯 Gymnasium 动作与状态空间定义
        # ==================================================
        self.action_space = spaces.Discrete(self.cfg.max_hubs * (self.N + 1))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, 3), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        """回合重置：随机下发基站，并返回状态和空字典(适配 Gymnasium V3)"""
        super().reset(seed=seed)
        self.hub_locations = np.random.choice(self.N, self.cfg.max_hubs, replace=False)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """生成神经网络的视觉输入 (State)"""
        obs = np.zeros((self.N, 3))
        obs[:, 0] = self.coords[:, 0] / self.cfg.map_size
        obs[:, 1] = self.coords[:, 1] / self.cfg.map_size
        for h_loc in self.hub_locations:
            if h_loc < self.N: 
                obs[h_loc, 2] = 1.0 
        return obs.astype(np.float32)

    def step(self, action):
        """核心交互函数：执行动作，模拟潮汐，结算奖励"""
        hub_idx = action // (self.N + 1)   
        target_node = action % (self.N + 1) 
        
        # 【拦截 1：非法动作防止无效游走 (已修复神级 Bug)】
        if target_node < self.N and target_node in self.hub_locations:
            if target_node == self.hub_locations[hub_idx]:
                # 💡 AI 决定“原地待命”(Stay)！直接 pass 放行，让它领取丰厚的阵型收益！
                pass
            else:
                # 💥 撞车了！想搬到【其他枢纽】占用的格子上，这才是无效游走。
                return self._get_obs(), -1.0, False, False, {"msg": "Collision", "avg_cost": 0, "avg_coverage": 0, "active_hubs": len([h for h in self.hub_locations if h < self.N])}
        
        # 执行有效搬迁
        self.hub_locations[hub_idx] = target_node
        active_hubs = [h for h in self.hub_locations if h < self.N]

        # 【拦截 2：防止消极怠工自杀现象】
        if not active_hubs:
            return self._get_obs(), -20.0, True, False, {"msg": "All hubs in trash!", "avg_cost": 0, "avg_coverage": 0, "active_hubs": 0}

        # ==================================================
        # 👑 精度与速度的控制台：每时段均匀抽 6 张，共 24 张
        # ==================================================
        samples_per_period = 6
        M = self.cfg.M_snapshots  # 每时段快照数 = 100
        T = self.cfg.T_periods    # 时段数 = 4
        sample_indices = []
        for t in range(T):
            period_start = t * M
            period_indices = np.random.choice(M, samples_per_period, replace=False) + period_start
            sample_indices.extend(period_indices.tolist())
        target_snapshots = [self.snapshots[i] for i in sample_indices]
        actual_sample_size = len(sample_indices)

        total_expected_op_cost = 0
        total_expected_coverage = 0
        total_expected_idle = 0
        
        # 💡 [新增]: 记录每张快照的综合成本，用于计算成本波动（标准差）
        snapshot_costs = []
        
        # 逐张快照打分
        for snap_demand in target_snapshots:
            op_cost, coverage, idle_p = self._sinkhorn_allocation(active_hubs, snap_demand)
            total_expected_op_cost += op_cost
            total_expected_coverage += coverage
            total_expected_idle += idle_p
            # 记录单张快照的纯运费+闲置税+超载罚款
            snapshot_costs.append(op_cost + idle_p)
        
        # 结算平均成绩
        avg_op_cost = total_expected_op_cost / actual_sample_size
        avg_coverage = total_expected_coverage / actual_sample_size
        avg_idle = total_expected_idle / actual_sample_size
        
        # 💡 [新增]: 计算跨时期成本波动的标准差
        std_op_cost = np.std(snapshot_costs)
        
        total_fixed_cost = sum([self.fixed_costs[h] for h in active_hubs])
        
        # 全新综合成本 = 建站费 + 平均运营费 + 闲置税 + λ * 跨期运营成本波动
        comprehensive_cost = total_fixed_cost + avg_op_cost + avg_idle + (self.lambda_robust * std_op_cost)

        # ==================================================
        # 💰 核心逻辑：归一化博弈奖励体系 (Reward Shaping)
        # ==================================================
        # 1. 恢复平滑的基础打分 (保证全空间的梯度连续性)
        cov_score = avg_coverage * 10.0
        
        normalized_cost = comprehensive_cost / self.baseline_cost
        
        # 引入 np.clip：强制截断惩罚
        safe_normalized_cost = np.clip(normalized_cost, 0, 2.0)
        cost_score = safe_normalized_cost * 10.0 # 成本权重
        
        base_reward = cov_score - cost_score
        
        # 2. 💡 [修复断崖]: 采用“连续加罚”而非“直接切断”
        if avg_coverage >= 0.95:
            # 安全区：只有基础收益，AI 会在这里安心压榨成本
            reward = base_reward
        else:
            # 危险区：在基础分上，额外扣除极其严厉的红线惩罚！
            # 覆盖率越低，扣得越狠，在数学上形成一个连续的“陡坡”，把 AI 逼回安全区
            penalty = 50.0 * (0.95 - avg_coverage)
            reward = base_reward - penalty
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= 100 # 每 100 步结束这一局

        info = {
            "avg_cost": comprehensive_cost,  
            "operational_cost": avg_op_cost,    
            "fixed_cost": total_fixed_cost,  
            "idle_penalty": avg_idle, 
            "std_op_cost": std_op_cost, # 加入字典，方便后续在 TensorBoard 里监控波动率下降       
            "avg_coverage": avg_coverage,
            "active_hubs": len(active_hubs),
            "hub_locations": active_hubs.copy() # 留给外界保存神级阵型使用
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _sinkhorn_allocation(self, active_hubs, snap_demand):
        """
        基于 Sinkhorn 变体的博弈分配：计算软分配期望值
        """
        num_hubs = len(active_hubs)
        
        # 1. 建立基础物理运费矩阵
        C_dist = np.zeros((self.N, num_hubs + 1))
        for j, h_idx in enumerate(active_hubs):
            distances = self.dist_matrix[:, h_idx]
            C_dist[:, j] = self.physics.calculate_economic_cost(distance=distances, demand=1.0)
            
        C_dist[:, -1] = self.cfg.penalty_unmet

        shadow_prices = np.zeros(num_hubs + 1)
        
        # 2. Sinkhorn 拥堵博弈迭代
        for _ in range(20):
            perceived_cost = C_dist + shadow_prices
            min_cost = np.min(perceived_cost, axis=1, keepdims=True)
            exp_cost = np.exp(-(perceived_cost - min_cost) / self.sinkhorn_temp)
            probs = exp_cost / np.sum(exp_cost, axis=1, keepdims=True)
            
            hub_loads = np.zeros(num_hubs)
            for j, h_idx in enumerate(active_hubs):
                mask = np.ones(self.N, dtype=bool)
                mask[h_idx] = False 
                hub_loads[j] = np.sum(probs[mask, j] * snap_demand[mask])
            
            overloads = np.maximum(0, hub_loads - self.cfg.Q)
            shadow_prices[:-1] += overloads * self.shadow_step

        # 3. 基于概率的期望结算 (保证梯度连续性)
        transport_cost = 0.0
        idle_penalty = 0.0 
        actual_hub_loads = np.zeros(num_hubs)

        for j, h_idx in enumerate(active_hubs):
            mask = np.ones(self.N, dtype=bool)
            mask[h_idx] = False # 屏蔽本地
            
            # 跨节点飞行的期望运费
            transport_cost += np.sum(probs[mask, j] * C_dist[mask, j] * snap_demand[mask])
            
            # 跨节点飞行占用的期望无人机载荷 (依然严谨屏蔽本地单)
            actual_hub_loads[j] = np.sum(probs[mask, j] * snap_demand[mask])
            
            # 💡【修正 P1：微观闲置税！深入到每个枢纽算闲置，拒绝一刀切】
            hub_idle = max(0, self.cfg.Q - actual_hub_loads[j])
            idle_penalty += hub_idle * (self.cfg.penalty_unmet * 0.005)

        # 违约金与超载罚金
        penalty_cost = np.sum(probs[:, -1] * snap_demand) * self.cfg.penalty_unmet
        soft_overloads = np.maximum(0, actual_hub_loads - self.cfg.Q)
        overload_penalty = np.sum(soft_overloads) * self.overload_coef 
        
        # 纯运营运费
        base_op_cost = transport_cost + penalty_cost + overload_penalty
        
        # 💡【终极修复 P2：100% 物理自洽的 Coverage】
        # 覆盖率 = 1.0 - (流向垃圾桶的需求 / 全图总需求)
        # 本地单天然被分配到本地基站(C_dist=0)，不占无人机，但也属于完美覆盖！绝对不会超过 1.0！
        coverage = 1.0 - np.sum(probs[:, -1] * snap_demand) / (np.sum(snap_demand) + 1e-6)

        # 返回三个值：纯运营成本, 覆盖率, 单独的闲置税
        return base_op_cost, coverage, idle_penalty