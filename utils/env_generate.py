import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import os
import pickle

# ==========================================
# 【核心寻址】确保能找到项目根目录
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig

def generate_and_save_data():
    cfg = UAVHubConfig()
    map_size = cfg.map_size          
    seed = cfg.seed                  
    np.random.seed(seed)
    
    print("=================================================")
    print(f"🌍 开始生成 [POMDP 连续日场景需求网络 - 严密时序版]")
    
    # ==========================================
    # 🌟 1~4. 3GPP 物理基座 (完全保留)
    # ==========================================
    block_size = 120.0   
    street_width = 20.0  
    cell_step = block_size + street_width 
    building_coverage_ratio = 0.5  
    city_open_area_ratio = 0.5     
    
    grid_dim = int(map_size / cell_step) 
    total_cells = grid_dim * grid_dim
    
    all_cells = []
    for gx in range(grid_dim):
        for gy in range(grid_dim):
            px = gx * cell_step + (street_width / 2.0) + (block_size / 2.0)
            py = gy * cell_step + (street_width / 2.0) + (block_size / 2.0)
            all_cells.append([px, py])
            
    all_cells = np.array(all_cells)
    np.random.shuffle(all_cells)
    num_building_cells = int(total_cells * (1 - city_open_area_ratio))
    
    coords = all_cells[:num_building_cells]  
    open_cells = all_cells[num_building_cells:]      
    actual_N = len(coords)
    
    print(f"🌍 物理尺寸: {map_size}x{map_size}m | 预留空地比: {int(city_open_area_ratio * 100)}%")
    print(f"🔥 全城共 {total_cells} 个网格，生成 {actual_N} 个活跃需求建筑。")

    heights = np.zeros(actual_N)       
    node_types = np.zeros(actual_N, dtype=int)  
    base_intensity = np.zeros(actual_N)
    
    num_commercial = int(actual_N * 0.3)
    shuffled_indices = np.random.permutation(actual_N)
    uma_indices = shuffled_indices[:num_commercial]
    umi_indices = shuffled_indices[num_commercial:]
    
    node_types[uma_indices] = 1 # UMa (商业)
    node_types[umi_indices] = 0 # UMi (住宅)

    print("🏢 正在注入服务节点 3GPP 标准高度与容积需求系数...")
    block_area = block_size * block_size 
    
    for i in uma_indices:
        heights[i] = np.random.uniform(20.0, 120.0) 
        floors = heights[i] / 3.0
        far = (block_area * building_coverage_ratio * floors) / block_area
        base_intensity[i] = far * 1.5  
        
    for i in umi_indices:
        heights[i] = np.random.uniform(9.0, 30.0)   
        floors = heights[i] / 3.0
        far = (block_area * building_coverage_ratio * floors) / block_area
        base_intensity[i] = far * 5.0  

    C = np.zeros((actual_N, actual_N))
    for i in range(actual_N):
        for j in range(actual_N):
            C[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    # 💡 [核心修复] 引入异质性建站成本
    f_min = getattr(cfg, 'f_min', 10000.0)
    f_max = getattr(cfg, 'f_max', 20000.0)
    
    # 按照建筑高度（容积率）来计算地价归一化系数
    height_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-8)
    
    # 高楼（商业区）贵，矮楼（住宅区）便宜
    f = f_min + height_norm * (f_max - f_min)
    
    # 加一点均匀分布的现实地价噪音 (±500)，防止被网络抓到纯线性漏洞
    noise = np.random.uniform(-500.0, 500.0, size=actual_N)
    f = np.clip(f + noise, f_min, f_max)
    # ==========================================
    # 🌟 5. 动态日场景生成 (修复时序残差与动态配置)
    # ==========================================
    T_timesteps = getattr(cfg, 'T_timesteps', 96)
    num_train_scenarios = getattr(cfg, 'num_train_scenarios', 100)
    num_eval_scenarios = getattr(cfg, 'num_eval_scenarios', 30) # 动态读取评估集大小
    sigma = getattr(cfg, 'tidal_sigma', 1.5)
    baseline = getattr(cfg, 'tidal_baseline', 0.1)
    peaks_UMa = getattr(cfg, 'tidal_peaks_UMa', [(8.0, 2.4), (13.0, 0.9)])
    peaks_UMi = getattr(cfg, 'tidal_peaks_UMi', [(13.0, 0.7), (18.0, 2.4)])

    def get_smooth_tidal_multipliers(hour):
        """完全由 config 驱动：每个节点类型的峰值列表独立叠加，无硬编码系数"""
        uma_val = baseline
        for peak_h, peak_mul in peaks_UMa:
            uma_val += peak_mul * np.exp(-0.5 * ((hour - peak_h) / sigma)**2)

        umi_val = baseline
        for peak_h, peak_mul in peaks_UMi:
            umi_val += peak_mul * np.exp(-0.5 * ((hour - peak_h) / sigma)**2)

        return uma_val, umi_val

    def generate_scenarios(num_days, dataset_name):
        print(f"🌊 正在生成 {dataset_name} 数据: {num_days} 个完整的日场景...")
        scenarios = []
        for d in range(num_days):
            day_data = np.zeros((T_timesteps, actual_N))
            day_busyness = np.random.uniform(0.7, 1.3)
            
            # AR(1) 自回归状态，用于模拟 15 分钟级的情绪惯性
            inertia_mult = 1.0 
            
            for t in range(T_timesteps):
                hour = t * 24.0 / T_timesteps
                mul_UMa, mul_UMi = get_smooth_tidal_multipliers(hour)
                
                # 更新 15 分钟级的时间残差惯性 (70% 继承上个槽的惯性，30% 产生新波动)
                inertia_mult = 0.7 * inertia_mult + 0.3 * np.random.normal(1.0, 0.4)
                inertia_mult = np.clip(inertia_mult, 0.5, 2.0) # 防止过度发散
                
                slot_duration_hours = 24.0 / T_timesteps 
                expected_demand = np.zeros(actual_N)
                
                # 回归纯粹的物理基准需求，不使用缩放器
                expected_demand[node_types == 1] = base_intensity[node_types == 1] * mul_UMa * slot_duration_hours * day_busyness * inertia_mult
                expected_demand[node_types == 0] = base_intensity[node_types == 0] * mul_UMi * slot_duration_hours * day_busyness * inertia_mult
                
                sampled_demand = np.random.poisson(expected_demand).astype(float)
                day_data[t] = sampled_demand * 1.2  # 乘以 1.2kg 的单均重量
                
            scenarios.append(day_data)
        return np.array(scenarios)

    # 分别生成训练集和评估集
    train_scenarios = generate_scenarios(num_train_scenarios, "【训练集】")
    eval_scenarios = generate_scenarios(num_eval_scenarios, "【评估集】")

    # ==========================================
    # 🌟 6. 数据打包与归档
    # ==========================================
    topo_data = {
        'coords': coords, 'heights': heights, 
        'obs_coords': np.array([]), 'obs_heights': np.array([]), 
        'open_cells': open_cells, 'C': C, 'f': f, 
        'node_types': node_types, 'base_intensity': base_intensity
    }

    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'map_adaptive_seed{seed}_robust.pkl')
    
    final_data = {
        'topo_data': topo_data,
        'daily_scenarios': train_scenarios, # 保留此 key 以防可视化脚本报错
        'train_scenarios': train_scenarios, # 明确分离训练集
        'eval_scenarios': eval_scenarios,   # 明确分离评估集
        'config_meta': {
            'map_size': map_size, 'block_size': block_size, 'street_width': street_width,
            'N': actual_N, 'seed': seed, 
            'T_timesteps': T_timesteps, 
            'num_train_scenarios': num_train_scenarios,
            'num_eval_scenarios': num_eval_scenarios
        }
    }
    with open(file_path, 'wb') as f_out:
        pickle.dump(final_data, f_out)

    # ==========================================
    # 💡 物理容量压力核算与配置建议
    # ==========================================
    max_train_peak = np.max(np.sum(train_scenarios, axis=2))
    current_Q = getattr(cfg, 'Q', 1500)
    total_capacity = current_Q * getattr(cfg, 'max_hubs', 3)
    
    print("-" * 50)
    print(f"🚨 [容量抗压核对] 全城 15 分钟并发需求峰值达: {max_train_peak:.1f} kg")
    print(f"   ➤ 当前 config.py 中 Q={current_Q}，全城总容量为 {total_capacity} kg。")
    
    if max_train_peak > total_capacity:
        print("   ➤ 状态：完美！峰值已击穿当前总容量，必定触发 RL 负载均衡！")
    else:
        # 给出科学的改参建议（容量设为峰值的 1/3 到 1/4 左右最为合适）
        suggested_Q = int((max_train_peak * 0.8) / getattr(cfg, 'max_hubs', 3) / 100) * 100 
        print(f"   ➤ 状态：过载压力不足！当前总容量 ({total_capacity} kg) 远大于需求峰值 ({max_train_peak:.1f} kg)。")
        print(f"   ➤ 建议：请前往 config.py，将 Q 值调低至约 【{suggested_Q}】，以保证在高峰期触发 20%~30% 的容量缺口！")
    print("-" * 50)
    print(f"✅ 数据生成完毕！已保存至: {file_path}")

if __name__ == '__main__':
    generate_and_save_data()