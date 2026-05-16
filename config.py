import numpy as np
from dataclasses import dataclass, field

@dataclass
class UAVHubConfig:
    # --- 0. 全局控制 ---
    seed: int = 42                 # 全局随机种子
    
    # --- 1. 无人机物理与设施参数 ---
    Q: float = 500.0              # 【已调低】单个枢纽的宏观承载容量 (kg) -> 故意制造容量焦虑
    E_max: float = 1200.0          # 电池最大可用能量 (Wh)
    e_full: float = 0.2            # 满载状态单位距离能耗系数 (Wh/m)
    e_empty: float = 0.1           # 空载状态单位距离能耗系数 (Wh/m)
    
    # --- 2. 地图与节点生成参数 ---
    map_size: float = 2000.0       # 地图边长 (m)
    f_min: float = 10000.0         # 建站成本下界
    f_max: float = 20000.0         # 建站成本上界
    
    # --- 🌟 3. 日场景与时间维度参数 (方案 B 核心) ---
    T_timesteps: int = 96          # 一天的分配时间槽数 (96个槽 = 24小时)
    slot_minutes: float = 15.0     # 每个时间槽对应的物理时间 (分钟)
    num_train_scenarios: int = 100 # 预生成训练的完整“日场景”数
    num_eval_scenarios: int = 30   # 评估用的“日场景”数

    # --- 🌟 4. 潮汐需求生成参数 (高斯混合平滑曲线) ---
    # 格式: [(峰值小时, 峰值相对基线的强度乘子), ...]
    # UMa (商业区): 早晨上班高峰 + 午间小高峰
    # UMi (住宅区): 晚间下班高峰 + 午间小高峰
    tidal_peaks_UMa: list = field(default_factory=lambda: [(8.0, 2.4), (13.0, 0.9)])
    tidal_peaks_UMi: list = field(default_factory=lambda: [(13.0, 0.7), (18.0, 2.4)])
    tidal_baseline: float = 0.1    # 深夜需求基线乘子
    tidal_sigma: float = 1.5       # 高斯峰的宽度（小时）
    
    # --- 5. 鲁棒选址与经济核算参数 ---
    max_hubs: int = 3              # [约束红线] 最多允许建设的枢纽数量
    
    # [灵魂参数] 惩罚金设定
    penalty_unmet: float = 100.0   # 拒单惩罚 (极高，逼迫 RL 在空间上做负载均衡，保留容量)
    penalty_wait: float = 5.0      # 推迟分配的等待惩罚 (为后续可能引入的延迟动作预留接口)