import numpy as np
from dataclasses import dataclass

@dataclass
class UAVHubConfig:
    # --- 0. 全局控制 ---
    seed: int = 42              # 全局随机种子
    
    # --- 1. 集合定义与基础参数 ---
    N: int = 20                 # 缩小场景：20个节点
    
    # --- 2. 无人机物理红线参数 (全牛顿 N 版本) ---
    Q: float = 200.0            # 单次最大载重量 (N) 
    E_max: float = 1200.0       # 电池最大可用能量 (Wh)
    e_full: float = 0.2         # 满载状态单位距离能耗系数 (Wh/m)
    e_empty: float = 0.1        # 空载状态单位距离能耗系数 (Wh/m)
    
    # --- 3. 地图与节点生成参数 ---
    map_size: float = 1000.0    # 地图边长 (m)
    f_min: float = 10000.0      # 建站成本下界
    f_max: float = 20000.0      # 建站成本上界
    
# --- 🌟 4. 潮汐需求与蒙特卡洛引擎参数 (严格物理边界版) ---
    M_snapshots: int = 100            # 每个时段生成的快照数量
    T_periods: int = 4                # 每天划分为4个时段 (早/午/晚/夜)

     # 【绝对物理边界控制】-> 留在这里，方便你随时调控考卷难度！
    range_low: tuple = (1, 14)         # 较小需求区间 (如夜间、非高峰的住宅区)
    range_normal: tuple = (15, 30)    # 正常需求区间
    range_surge: tuple = (30, 45)     # 爆单需求区间 (死死卡住上限50，绝不打穿无人机运力)
    
    # --- 5. 鲁棒选址与经济核算核心参数 ---
    max_hubs: int = 5           # [约束红线] 最多允许建设的枢纽数量
    
    # [灵魂参数] 单位重量拒单惩罚金 (每丢 1N 的需求扣多少分)
    # 必须显著高于运输成本，逼迫 RL 留出运力冗余。
    penalty_unmet: float = 100.0