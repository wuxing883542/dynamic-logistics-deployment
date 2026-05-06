import numpy as np
import os
import pickle
import sys

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
    N = cfg.N
    seed = cfg.seed
    np.random.seed(seed)
    
    print("=================================================")
    print(f"🌍 开始生成 [带隔离带的严格栅格化 3D 城市底座]")
    
    # ==========================================
    # 🌟 1. 带“隔离带”的网格分配引擎 (杜绝贴靠)
    # ==========================================
    grid_res = 50.0  
    cells_per_axis = int(cfg.map_size / grid_res)  # 20x20 = 400 个格子
    center = cfg.map_size / 2.0
    
    grid_occupied = np.zeros((cells_per_axis, cells_per_axis), dtype=bool)
    
    def get_free_cell(min_r, max_r):
        for _ in range(2000): # 最多尝试 2000 次
            gx = np.random.randint(0, cells_per_axis)
            gy = np.random.randint(0, cells_per_axis)
            px = gx * grid_res + grid_res / 2
            py = gy * grid_res + grid_res / 2
            
            dist = np.linalg.norm([px - center, py - center])
            if min_r <= dist <= max_r:
                # 检查自身及周围 8 个邻居是否被占用 (3x3 安全隔离带)
                x_min, x_max = max(0, gx-1), min(cells_per_axis, gx+2)
                y_min, y_max = max(0, gy-1), min(cells_per_axis, gy+2)
                
                if not np.any(grid_occupied[x_min:x_max, y_min:y_max]):
                    grid_occupied[gx, gy] = True 
                    return px, py
        raise ValueError("地图太拥挤，无法找到符合隔离要求的空地！请减少建筑物数量。")

    coords = np.zeros((N, 2))
    heights = np.zeros(N)       
    node_types = np.zeros(N, dtype=int)  
    
    num_commercial = int(N * 0.3) 
    num_residential = N - num_commercial 
    
    # 【分配商业区】市中心 (半径 0 ~ 250m)
    for i in range(num_commercial):
        coords[i] = get_free_cell(0, 250)
        heights[i] = np.random.uniform(50.0, 120.0)
        node_types[i] = 1
        
    # 【分配住宅区】外围 (半径 200m ~ 450m)
    for i in range(num_residential):
        idx = num_commercial + i
        coords[idx] = get_free_cell(200, 450)
        heights[idx] = np.random.uniform(10.0, 30.0)
        node_types[idx] = 0
        
    # ==========================================
    # 🌟 1.5 分层分配自然遮挡 (均匀分布在市中心和郊区)
    # ==========================================
    num_obstacles = getattr(cfg, 'num_obstacles', 15) 
    obs_coords = np.zeros((num_obstacles, 2))
    obs_heights = np.zeros(num_obstacles)
    
    # 刻意切分：约 30% 放市中心当市政公园/塔，剩下的放郊区
    num_center_obs = int(num_obstacles * 0.3)
    
    for i in range(num_center_obs):
        # 强制塞进 CBD 核心区 (半径 0~250)
        obs_coords[i] = get_free_cell(0, 250)
        obs_heights[i] = np.random.uniform(15.0, 45.0) 
        
    for i in range(num_center_obs, num_obstacles):
        # 剩下的丢在郊区 (半径 250~500)
        obs_coords[i] = get_free_cell(250, 500)
        obs_heights[i] = np.random.uniform(15.0, 45.0) 

    # ==========================================
    # 🌟 2. 距离与潮汐快照计算 
    # ==========================================
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    f = np.full(N, getattr(cfg, 'f_min', 10000.0))

    range_low = getattr(cfg, 'range_low', (1, 14))
    range_normal = getattr(cfg, 'range_normal', (15, 30))
    range_surge = getattr(cfg, 'range_surge', (30, 45))
    
    T_periods = getattr(cfg, 'T_periods', 4)
    M_snapshots = getattr(cfg, 'M_snapshots', 100)
    num_total_snapshots = M_snapshots * T_periods
    snapshots_total = np.zeros((num_total_snapshots, N))
    
    tidal_states = [
        (range_surge, range_low),       
        (range_low, range_surge),       
        (range_normal, range_normal),   
        (range_low, range_low)          
    ]
    
    snapshot_idx = 0
    for t in range(T_periods):
        state_R, state_C = tidal_states[t]
        for m in range(M_snapshots):
            demand_m = np.zeros(N)
            demand_C = np.random.uniform(state_C[0], state_C[1], num_commercial)
            demand_m[node_types == 1] = demand_C
            demand_R = np.random.uniform(state_R[0], state_R[1], num_residential)
            demand_m[node_types == 0] = demand_R
            snapshots_total[snapshot_idx] = demand_m
            snapshot_idx += 1
    
    # 打包存档
    topo_data = {
        'coords': coords,            
        'heights': heights,          
        'obs_coords': obs_coords,    
        'obs_heights': obs_heights,  
        'C': C,
        'f': f,
        'node_types': node_types
    }

    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'map_{N}n_seed{seed}_robust.pkl')
    
    final_data = {
        'topo_data': topo_data,
        'snapshots_total': snapshots_total,
        'config_meta': {'N': N, 'seed': seed, 'M_snapshots': M_snapshots, 'T_periods': T_periods}
    }
    with open(file_path, 'wb') as f_out:
        pickle.dump(final_data, f_out)

    print(f"✅ 带隔离带的网格化底座生成完毕，中心与边缘遮挡分布已优化！")

if __name__ == '__main__':
    generate_and_save_data()