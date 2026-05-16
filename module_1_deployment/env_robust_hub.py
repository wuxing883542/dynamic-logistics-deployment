import sys
import os
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig


class RobustHubEnv(gym.Env):
    """
    Option B: 选址 + 时序调度 双阶段 PPO 环境

    Phase 1 (t=0, site_selection):
        action = (K,) int array ─ K 个枢纽节点索引, 无重复
        reward = -fixed_cost / 100

    Phase 2 (t=1..T, dispatch):
        action = (N,) int array ─ 每个节点的分配决策:
            0..K-1 → 分配给对应枢纽
            K      → 拒单
        reward = -(transport_cost + unmet_penalty) / 100

    Hub 容量 (Q) 每调度步重置 (per-slot 语义).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: UAVHubConfig, mode: str = 'train'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        # ── 加载数据 ──
        data_path = os.path.join(BASE_DIR, 'data', f'map_adaptive_seed{cfg.seed}_robust.pkl')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}. Run utils/env_generate.py first."
            )

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.topo = data['topo_data']
        self.train_scenarios = data['train_scenarios']   # (100, 96, N)
        self.eval_scenarios = data.get('eval_scenarios', data['train_scenarios'][-30:])
        self.N = data['config_meta']['N']
        self.T = cfg.T_timesteps
        self.K = cfg.max_hubs

        self.coords = self.topo['coords']
        self.dist_matrix = self.topo['C']
        self.fixed_costs = self.topo['f']
        self.node_types = self.topo['node_types']

        # ── 状态变量 ──
        self.current_scenario = None   # (T, N) 当前日场景
        self.current_t = 0             # 当前调度步 (0..T-1)
        self.hub_locations = []        # K 个枢纽索引
        self.hub_capacities = np.zeros(self.K)
        self.done_site_selection = False

        # ── 动作空间 (Gym 兼容占位, 真实格式见 step 文档) ──
        self.action_space = spaces.Box(
            low=0, high=max(self.N, self.K), shape=(self.N,), dtype=np.float32
        )

        # ── 观测空间 ──
        self.observation_space = spaces.Dict({
            'phase':           spaces.Discrete(2),
            'node_features':   spaces.Box(low=-1.0, high=1.0, shape=(self.N, 5), dtype=np.float32),
            'current_orders':  spaces.Box(low=0.0, high=500.0, shape=(self.N,), dtype=np.float32),
            'hub_mask':        spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            'hub_capacities':  spaces.Box(low=0.0, high=1.0, shape=(self.K,), dtype=np.float32),
        })

    # ── Reset ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        scenarios = self.train_scenarios if self.mode == 'train' else self.eval_scenarios
        day_idx = self.np_random.integers(len(scenarios))
        self.current_scenario = scenarios[day_idx]

        self.current_t = 0
        self.hub_locations = []
        self.hub_capacities = np.full(self.K, self.cfg.Q)
        self.done_site_selection = False

        return self._get_obs(), {}

    # ── 观测构建 ───────────────────────────────────────────
    def _get_obs(self):
        node_f = np.zeros((self.N, 5), dtype=np.float32)
        node_f[:, 0] = self.coords[:, 0] / self.cfg.map_size          # x 归一化
        node_f[:, 1] = self.coords[:, 1] / self.cfg.map_size          # y 归一化
        node_f[:, 2] = self.node_types.astype(np.float32)             # 0=UMi, 1=UMa
        node_f[:, 3] = self.fixed_costs / self.cfg.f_max              # 建站成本归一化

        center = np.array([self.cfg.map_size * 0.5, self.cfg.map_size * 0.5])
        node_f[:, 4] = np.linalg.norm(self.coords - center, axis=1) / (self.cfg.map_size * 0.707)

        hub_mask = np.zeros(self.N, dtype=np.float32)
        for h in self.hub_locations:
            hub_mask[h] = 1.0

        orders = np.zeros(self.N, dtype=np.float32)
        if self.done_site_selection and self.current_t < self.T:
            orders = self.current_scenario[self.current_t].astype(np.float32)

        return {
            'phase':          0 if not self.done_site_selection else 1,
            'node_features':  node_f,
            'current_orders': orders,
            'hub_mask':       hub_mask,
            'hub_capacities': self.hub_capacities.copy() / max(self.cfg.Q, 1.0),
        }

    # ── Step ───────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.int32).flatten()

        # ===================================================
        # Phase 1: 选址 (t=0, 只执行一次)
        # ===================================================
        if not self.done_site_selection:
            if len(action) < self.K:
                raise ValueError(
                    f"Site selection needs {self.K} hub indices, got {len(action)}"
                )

            hubs = [int(action[i]) for i in range(self.K)]

            for h in hubs:
                if h < 0 or h >= self.N:
                    raise ValueError(f"Hub index {h} out of range [0, {self.N - 1}]")

            if len(set(hubs)) < self.K:
                raise ValueError(
                    f"Duplicate hub indices in site selection: {hubs}. "
                    f"Plackett-Luce must produce unique indices."
                )

            self.hub_locations = hubs
            self.done_site_selection = True
            self.current_t = 0

            fixed_cost = sum(self.fixed_costs[h] for h in hubs)
            reward = -fixed_cost / 100.0

            info = {
                'phase':      'site_selection',
                'hubs':       hubs.copy(),
                'fixed_cost': fixed_cost,
            }
            return self._get_obs(), reward, False, False, info

        # ===================================================
        # Phase 2: 调度 (t = 1 .. T, 共 T 步)
        # ===================================================
        orders = self.current_scenario[self.current_t]
        active_nodes = np.where(orders > 0)[0]

        # 每调度步重置容量 (Q = per-slot 语义)
        self.hub_capacities = np.full(self.K, self.cfg.Q)

        transport_cost = 0.0
        unmet_penalty = 0.0
        served_per_hub = np.zeros(self.K)
        per_node_cost = np.zeros(self.N)

        # 按需求量降序处理, 减少碎片化
        sorted_idx = np.argsort(-orders[active_nodes])

        for idx in sorted_idx:
            i = active_nodes[idx]
            demand = orders[i]

            hub_choice = int(action[i]) if i < len(action) else self.K

            if hub_choice < 0 or hub_choice >= self.K:
                unmet_penalty += demand * self.cfg.penalty_unmet
                per_node_cost[i] = demand * self.cfg.penalty_unmet
            else:
                dist = self.dist_matrix[i, self.hub_locations[hub_choice]]
                allocated = min(demand, self.hub_capacities[hub_choice])

                transport_cost += allocated * dist * 0.01
                self.hub_capacities[hub_choice] -= allocated
                served_per_hub[hub_choice] += allocated

                if allocated < demand:
                    unmet_penalty += (demand - allocated) * self.cfg.penalty_unmet

                per_node_cost[i] = allocated * dist * 0.01 + (demand - allocated) * self.cfg.penalty_unmet

        reward = -(transport_cost + unmet_penalty) / 100.0

        self.current_t += 1
        terminated = self.current_t >= self.T

        info = {
            'phase':           'dispatch',
            't':               self.current_t,
            'transport_cost':  transport_cost,
            'unmet_penalty':   unmet_penalty,
            'step_cost':       transport_cost + unmet_penalty,
            'served_per_hub':  served_per_hub.tolist(),
            'active_orders':   len(active_nodes),
            'total_demand':    float(orders.sum()),
            'per_node_cost':   per_node_cost,  # (N,) per-node cost in original currency
        }

        return self._get_obs(), reward, terminated, False, info

    def get_action_mask(self):
        """返回当前阶段的有效动作 Mask。

        Phase 0 (选址): (N,) bool — 所有节点均可选为枢纽
        Phase 1 (调度): (N, K+1) bool — N 个节点各 K+1 个选项 (枢纽0..K-1 + 拒单)
        """
        if not self.done_site_selection:
            return np.ones(self.N, dtype=bool)
        else:
            return np.ones((self.N, self.K + 1), dtype=bool)

