import numpy as np

class UAVPhysicsModel:
    """
    无人机物理模型类
    负责执行所有与无人机硬件限制、能耗计算相关的红线校验。
    """
    def __init__(self, config):
        """
        初始化物理模型
        :param config: 包含 e_full, e_empty, E_max, Q 等参数的配置对象
        """
        self.cfg = config

    def check_energy_red_line(self, distances):
        """
        【星型往返能耗保底校验】
        原理：
        1. 假设无人机针对片区内每个节点都采用“送一趟回一趟”的最笨策略。
        2. 去程假设为满载 (e_full)，回程假设为空载 (e_empty)。
        3. 如果这种极端情况下的总能耗 Σ [(e_full + e_empty) * d_ik] 仍小于电池容量 E_max，
           则第二阶段的 VRP 环形路径优化具有 100% 的可行性保底。
        
        :param distances: 节点到对应枢纽的距离列表或数组 (m)
        :return: bool, True表示安全，False表示超能耗
        """
        if len(distances) == 0:
            return True
        
        # 强制转换为 numpy 数组以支持向量化运算
        dist_arr = np.array(distances)
        
        # 单位距离往返能耗系数 (Wh/m)
        unit_trip_energy = self.cfg.e_full + self.cfg.e_empty
        
        # 计算该片区内所有节点的往返总能耗
        total_energy_consumed = np.sum(unit_trip_energy * dist_arr)
        
        return total_energy_consumed <= self.cfg.E_max

    def check_payload_red_line(self, total_demand):
        """
        【载重红线校验】
        确保该枢纽片区内的总需求量不超过无人机的最大额定载荷 Q 。
        这是为了保证无人机一次起飞能带走该片区所有的货物。
        
        :param total_demand: 片区总需求量 (N)
        :return: bool
        """
        return total_demand <= self.cfg.Q

    def check_capacity_constraint(self, total_demand, capacity_limit):
        """
        【枢纽库容校验】
        确保分配给该枢纽的货物总量不超过其库存容量上限 lambda_cap。
        
        :param total_demand: 分配给该枢纽的总需求量
        :param capacity_limit: 枢纽的容量上限
        :return: bool
        """
        return total_demand <= capacity_limit

    def calculate_economic_cost(self, distance, demand):
        """
        【经济成本计算】
        用于强化学习的奖励函数设计。反映运输过程中的周转成本。
        通常公式为：运输成本 = 距离 * 需求量
        
        :param distance: 节点与枢纽间的距离 (m)
        :param demand: 节点的需求量 (N)
        :return: float, 运输代价
        """
        unit_cost = 0.01#经济成本折算系数
        return distance * demand *unit_cost