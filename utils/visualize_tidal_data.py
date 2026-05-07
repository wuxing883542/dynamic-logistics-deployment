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
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    topo_data = data['topo_data']
    snapshots = data['snapshots_total']
    meta = data['config_meta']

    coords = topo_data['coords']
    heights = topo_data['heights']
    node_types = topo_data['node_types'] 
    obs_coords = topo_data.get('obs_coords', np.array([]))
    obs_heights = topo_data.get('obs_heights', np.array([]))

    T, M = meta['T_periods'], meta['M_snapshots']

    # ==========================================
    # 🚀 全局数据抗压统计监控打印
    # ==========================================
    global_max_node = np.max(snapshots)
    global_min_node = np.min(snapshots)
    snapshot_totals = np.sum(snapshots, axis=1) 
    max_total_demand = np.max(snapshot_totals)
    min_total_demand = np.min(snapshot_totals)

    print("-" * 50)
    print(f"📊 全局数据抗压统计监控:")
    print(f"   ➤ [局部极限] 所有快照中，单节点最大爆发量: {global_max_node:.2f}")
    print(f"   ➤ [局部极限] 所有快照中，单节点最小沉寂量: {global_min_node:.2f}")
    print(f"   ➤ [全城并发] 所有快照中，全城并发总需求最大峰值: {max_total_demand:.2f}")
    print(f"   ➤ [全城并发] 所有快照中，全城并发总需求最小谷值: {min_total_demand:.2f}")
    print("-" * 50)

    # ==========================================
    # 颜色与高度映射设置
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    all_heights = np.concatenate([heights, obs_heights]) if len(obs_heights) > 0 else heights
    norm = plt.Normalize(vmin=0, vmax=np.max(all_heights) + 10) 
    cmap = plt.cm.Blues 

    # 升级版绘制方块函数 (传入指定的 ax)
    def draw_grid_block(ax, cx, cy, h, edge_color, line_width, line_style='-'):
        bl_x, bl_y = cx - 25, cy - 25 
        face_color = cmap(norm(h))
        
        rect = patches.Rectangle((bl_x, bl_y), 50, 50, 
                                 linewidth=line_width, edgecolor=edge_color, 
                                 facecolor=face_color, linestyle=line_style, zorder=2)
        ax.add_patch(rect)
        
        text_color = 'white' if norm(h) > 0.5 else 'black'
        ax.text(cx, cy, f"{int(h)}m", color=text_color, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

    # 网格线设置函数
    def setup_grid(ax):
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        grid_ticks = np.arange(0, 1001, 50)
        ax.set_xticks(grid_ticks)
        ax.set_yticks(grid_ticks)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([str(x) if x % 200 == 0 else '' for x in grid_ticks])
        ax.set_yticklabels([str(y) if y % 200 == 0 else '' for y in grid_ticks])
        ax.set_xlabel("X 坐标 (米)")
        ax.set_ylabel("Y 坐标 (米)")

    # ==========================================
    # 🌟 第一部分：原版带双子图的完整视图
    # ==========================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
    ax1 = axes1[0]
    setup_grid(ax1)

    # 1. 绘制无需求遮挡物 (灰色虚线边框)
    if len(obs_coords) > 0:
        for (x, y), h in zip(obs_coords, obs_heights):
            draw_grid_block(ax1, x, y, h, 'dimgray', 2, '--')

    # 2. 绘制住宅区 (明绿色实线边框)
    R_mask = node_types == 0
    for (x, y), h in zip(coords[R_mask], heights[R_mask]):
        draw_grid_block(ax1, x, y, h, 'limegreen', 3)

    # 3. 绘制商业区 (红色实线边框)
    C_mask = node_types == 1
    for (x, y), h in zip(coords[C_mask], heights[C_mask]):
        draw_grid_block(ax1, x, y, h, 'red', 3)

    ax1.set_title("3D 城市建筑网格 (包含自然遮挡)", fontsize=14, fontweight='bold')
    legend_elements1 = [
        patches.Patch(facecolor='white', edgecolor='red', linewidth=3, label='商业大厦 (C)'),
        patches.Patch(facecolor='white', edgecolor='limegreen', linewidth=3, label='住宅建筑 (R)'),
        patches.Patch(facecolor='white', edgecolor='dimgray', linewidth=2, linestyle='--', label='自然遮挡 (无需求)')
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', bbox_to_anchor=(1.05, 1.15))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig1.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04).set_label('建筑 Z 轴高度 (米)', fontsize=12, fontweight='bold')

    # 子图 1.2: 潮汐波峰验证
    ax2 = axes1[1]
    periods = ['早高峰', '午高峰', '晚高峰', '深夜平峰']
    avg_R, avg_C = [], []
    for t in range(T):
        snaps = snapshots[t*M : (t+1)*M] 
        avg_R.append(np.mean(snaps[:, node_types == 0]))
        avg_C.append(np.mean(snaps[:, node_types == 1]))

    x = np.arange(len(periods))
    width = 0.35
    ax2.bar(x - width/2, avg_R, width, label='住宅区', color='limegreen')
    ax2.bar(x + width/2, avg_C, width, label='商业区', color='red')
    ax2.set_title("各时段潮汐需求期望波动", fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path1 = os.path.join(data_dir, pkl_file_name.replace('.pkl', '_UnifiedGrid_Heights.png'))
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    
    # ==========================================
    # 🌟 第二部分：额外生成单张【无遮挡】需求网格图
    # ==========================================
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    setup_grid(ax3)

    # 仅绘制住宅区和商业区，跳过自然遮挡物
    for (x, y), h in zip(coords[R_mask], heights[R_mask]):
        draw_grid_block(ax3, x, y, h, 'limegreen', 3)
    for (x, y), h in zip(coords[C_mask], heights[C_mask]):
        draw_grid_block(ax3, x, y, h, 'red', 3)

    ax3.set_title("3D 城市建筑网格 (仅需求节点)", fontsize=14, fontweight='bold')
    legend_elements2 = [
        patches.Patch(facecolor='white', edgecolor='red', linewidth=3, label='商业大厦 (C)'),
        patches.Patch(facecolor='white', edgecolor='limegreen', linewidth=3, label='住宅建筑 (R)')
    ]
    ax3.legend(handles=legend_elements2, loc='upper right', bbox_to_anchor=(1.05, 1.15))
    fig2.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04).set_label('建筑 Z 轴高度 (米)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path2 = os.path.join(data_dir, pkl_file_name.replace('.pkl', '_NoObsGrid_Heights.png'))
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    
    print(f"✅ 图片 1 (完整版) 已保存至: {save_path1}")
    print(f"✅ 图片 2 (无遮挡版) 已保存至: {save_path2}")
    
    plt.show() 

if __name__ == '__main__':
    test_file = "map_20n_seed42_robust.pkl" 
    visualize_tidal_data(test_file)