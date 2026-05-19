import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_tidal_data(pkl_file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    file_path = os.path.join(data_dir, pkl_file_name)
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    topo_data = data['topo_data']
    scenarios = data['daily_scenarios']
    meta = data['config_meta']

    coords = topo_data['coords']
    heights = topo_data['heights']
    node_types = topo_data['node_types'] 

    T_timesteps = meta.get('T_timesteps', 96)
    num_days = meta.get('num_train_scenarios', 100)
    block_size = meta.get('block_size', 120.0)
    street_width = meta.get('street_width', 20.0)
    half_b = block_size / 2.0
    map_size = meta.get('map_size', 2000.0)
    K = meta.get('max_hubs', 3) # 尝试获取配置中的枢纽数量，默认为 3

    # ==========================================
    # 🚀 全局数据抗压统计监控 & 运力 Q 验证
    # ==========================================
    global_max_node = np.max(scenarios)
    global_min_node = np.min(scenarios)
    
    # 单步 (15分钟) 全城并发统计
    snapshot_totals = np.sum(scenarios, axis=2) 
    max_step_demand = np.max(snapshot_totals)
    min_step_demand = np.min(snapshot_totals)

    # 💡 核心改动：全天全城总需求统计 (适配最新 MDP 架构)
    daily_totals = np.sum(scenarios, axis=(1, 2))
    max_daily_demand = np.max(daily_totals)
    mean_daily_demand = np.mean(daily_totals)

    print("-" * 65)
    print(f"📊 连续日场景 (POMDP) 数据抗压监控 & 全天运力(Q)设置参考:")
    print(f"   ➤ [数据规模] 共有 {num_days} 天数据，每天 {T_timesteps} 个时间槽")
    print(f"   ➤ [单步局部] 15分钟内，单节点最大爆发: {global_max_node:.2f} N")
    print(f"   ➤ [单步全城] 15分钟内，全城最大并发峰值: {max_step_demand:.2f} N")
    print(f"   ➤ [全天大盘] 全城单日平均总需求: {mean_daily_demand:.2f} N")
    print(f"   ➤ [全天大盘] 全城单日最高总需求: {max_daily_demand:.2f} N")
    print(f"   💡 [参数设置建议] 假设系统配置了 K={K} 个枢纽：")
    print(f"      - 宽裕运力配置 Q ≈ {max_daily_demand / K * 1.1:.0f} (满足极限爆单，毫无压力)")
    print(f"      - 紧凑运筹配置 Q ≈ {mean_daily_demand / K * 0.9:.0f} (强烈推荐！逼迫AI结合LSTM学会囤积与取舍)")
    print("-" * 65)

    # ==========================================
    # 样式配置
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    norm = plt.Normalize(vmin=0, vmax=np.max(heights) + 10) 
    cmap = plt.cm.Blues 

    def draw_grid_block(ax, cx, cy, h, edge_color, line_width):
        bl_x, bl_y = cx - half_b, cy - half_b 
        face_color = cmap(norm(h))
        rect = patches.Rectangle((bl_x, bl_y), block_size, block_size, 
                                 linewidth=line_width, edgecolor=edge_color, 
                                 facecolor=face_color, zorder=2)
        ax.add_patch(rect)
        text_color = 'white' if norm(h) > 0.5 else 'black'
        # 写高度 (带 m)
        ax.text(cx, cy, f"{int(h)}m", color=text_color, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # ==========================================
    # 🌟 图 1：曼哈顿网格拓扑 (高度图)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(11, 8.5))
    fig1.subplots_adjust(top=0.85)
    
    ax1.set_xlim(0, map_size)
    ax1.set_ylim(0, map_size)
    cell_step = block_size + street_width
    grid_ticks = np.arange(0, int(map_size) + 1, cell_step * 2) 
    ax1.set_xticks(grid_ticks)
    ax1.set_yticks(grid_ticks)
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    R_mask = node_types == 0
    for (x, y), h in zip(coords[R_mask], heights[R_mask]):
        draw_grid_block(ax1, x, y, h, 'limegreen', 2.5)

    C_mask = node_types == 1
    for (x, y), h in zip(coords[C_mask], heights[C_mask]):
        draw_grid_block(ax1, x, y, h, 'red', 2.5)

    ax1.set_title("3GPP 曼哈顿网格 (活跃服务节点拓扑 - 高度)", fontsize=16, fontweight='bold', pad=35)
    ax1.set_xlabel("X 坐标 (米)", fontsize=12)
    ax1.set_ylabel("Y 坐标 (米)", fontsize=12)
    
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='red', linewidth=3, label='UMa 商业区'),
        patches.Patch(facecolor='white', edgecolor='limegreen', linewidth=3, label='UMi 住宅区')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=12, framealpha=1.0)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar1 = fig1.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('建筑 Z 轴物理高度 (米)', fontsize=13)

    save_path1 = os.path.join(data_dir, pkl_file_name.replace('.pkl', '_Topology.png'))
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')

    # ==========================================
    # 🌟 图 2：24小时连续潮汐曲线 
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(12, 6.5))
    fig2.subplots_adjust(top=0.85)
    
    avg_day = np.mean(scenarios, axis=0) 
    avg_R = np.mean(avg_day[:, node_types == 0], axis=1) 
    avg_C = np.mean(avg_day[:, node_types == 1], axis=1) 
    
    hours = np.linspace(0, 24, T_timesteps, endpoint=False)
    
    ax2.plot(hours, avg_R, label='UMi 住宅区平均需求', color='limegreen', linewidth=3.5, zorder=5)
    ax2.plot(hours, avg_C, label='UMa 商业区平均需求', color='red', linewidth=3.5, zorder=5)
    
    ax2.set_title("全天候高斯平滑潮汐曲线 (100天均值)", fontsize=16, fontweight='bold', pad=35)
    ax2.set_xlabel("一天中的时间 (小时)", fontsize=13)
    ax2.set_ylabel("平均单节点需求重量 (N / 15分钟)", fontsize=13)
    
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.set_xticklabels([f"{int(h)}:00" for h in np.arange(0, 25, 2)])
    ax2.set_xlim(0, 24)
    
    ax2.legend(loc='upper right', fontsize=12, framealpha=1.0, edgecolor='gray')
    ax2.grid(True, linestyle=':', alpha=0.6)

    save_path2 = os.path.join(data_dir, pkl_file_name.replace('.pkl', '_TidalCurve.png'))
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')

    # ==========================================
    # 🌟 图 3：曼哈顿网格拓扑 (带节点编号 ID版 - RL 调参必备)
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(11, 8.5))
    fig3.subplots_adjust(top=0.85)
    
    ax3.set_xlim(0, map_size)
    ax3.set_ylim(0, map_size)
    ax3.set_xticks(grid_ticks)
    ax3.set_yticks(grid_ticks)
    ax3.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # 遍历所有节点，标上真实的编号 ID (0 到 N-1)
    for i, (x, y) in enumerate(coords):
        h = heights[i]
        n_type = node_types[i]
        
        edge_color = 'red' if n_type == 1 else 'limegreen'
        bl_x, bl_y = x - half_b, y - half_b 
        face_color = cmap(norm(h))
        
        rect = patches.Rectangle((bl_x, bl_y), block_size, block_size, 
                                 linewidth=2.5, edgecolor=edge_color, 
                                 facecolor=face_color, zorder=2)
        ax3.add_patch(rect)
        
        # 💡 核心改动：这里不写高度，写节点索引 i！字号调大一点方便查看
        text_color = 'white' if norm(h) > 0.5 else 'black'
        ax3.text(x, y, str(i), color=text_color, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

    ax3.set_title("3GPP 曼哈顿网格 (带节点编号 ID - RL 调参比对专用)", fontsize=16, fontweight='bold', pad=35)
    ax3.set_xlabel("X 坐标 (米)", fontsize=12)
    ax3.set_ylabel("Y 坐标 (米)", fontsize=12)
    
    ax3.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=12, framealpha=1.0)
    
    cbar3 = fig3.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('背景色依然代表建筑高度', fontsize=13)

    save_path3 = os.path.join(data_dir, pkl_file_name.replace('.pkl', '_Topology_IDs.png'))
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')

    # ==========================================
    # 打印保存信息并展示
    # ==========================================
    print(f"✅ 地图高度拓扑已保存至: {save_path1}")
    print(f"✅ 潮汐曲线已保存至: {save_path2}")
    print(f"✅ 【新增】带节点编号地图已保存至: {save_path3}")
    
    plt.show() 

if __name__ == '__main__':
    test_file = "map_adaptive_seed42_robust.pkl" 
    visualize_tidal_data(test_file)