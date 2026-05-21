import numpy as np
from dataclasses import dataclass, field

@dataclass
class UAVHubConfig:
    # --- 0. 全局控制 ---
    seed: int = 42                 # 全局随机种子
    
    # --- 1. 宏观枢纽与约束参数 (单位: N ) ---
    Q: float = 9000.0               # 单个枢纽每天最大的处理容量 (单位 N)
    max_flight_radius: float = 850.0 # 枢纽最大服务半径(m)。超出此范围的订单无法分配给该枢纽
    max_hubs: int = 3              # 全城固定的枢纽数量 (不再需要选址，稍后在环境里直接固定坐标)

    # --- 2. 城市空间参数 ---
    map_size: float = 2000.0       # 地图边长 (m)
    
    # --- 3. 日场景与时间维度参数 (POMDP 核心) ---
    T_timesteps: int = 96          # 一天的分配时间槽数 (96个槽 = 24小时)
    slot_minutes: float = 15.0     # 每个时间槽对应的物理时间 (分钟)
    num_train_scenarios: int = 100 # 预生成训练的完整“日场景”数
    num_eval_scenarios: int = 30   # 评估用的“日场景”数

    # --- 4. 潮汐需求生成参数 (高斯混合平滑曲线) ---
    # 格式: [(峰值小时, 峰值相对基线的强度乘子), ...]
    tidal_peaks_UMa: list = field(default_factory=lambda: [(8.0, 2.4), (13.0, 0.9)])
    tidal_peaks_UMi: list = field(default_factory=lambda: [(13.0, 0.7), (18.0, 2.4)])
    tidal_baseline: float = 0.1    # 深夜需求基线乘子
    tidal_sigma: float = 1.5       # 高斯峰的宽度（小时）
    
    # --- 5. 强化学习调度惩罚机制 ---
    penalty_unmet: float = 100.0   # 拒单惩罚