import os
import pickle
import numpy as np

def inspect_single_snapshots():
    """
    微观视角：抽查单张快照，验证真实波动与零需求 (p_zero) 机制
    自动寻找 data 目录下的 pkl 文件并自适应读取
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # 自动寻找 data 目录下的第一个 .pkl 文件
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if not pkl_files:
        print("❌ 错误：在 data/ 目录下没有找到任何 .pkl 文件！请先运行 env_generate.py")
        return
        
    target_file = pkl_files[0]
    file_path = os.path.join(data_dir, target_file)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 自适应键名解析（兼容你的旧版和我的修改版）
    snapshots = data.get('snapshots_total', data.get('snapshots'))
    topo = data.get('topo_data', data.get('topo'))
    node_types = topo['node_types']  

    idx_R = np.where(node_types == 0)[0]
    idx_C = np.where(node_types == 1)[0]

    test_indices = {
        "早高峰 (Snapshot #5)": 5,
        "午高峰 (Snapshot #105)": 105
    }

    print("=================================================")
    print(f"🔍 成功读取文件: {target_file}")
    print("=================================================")

    for period_name, snap_idx in test_indices.items():
        single_snapshot = snapshots[snap_idx]
        print(f"\n🕒 时段: {period_name}")
        
        # 打印住宅区
        demand_R = single_snapshot[idx_R]
        zeros_R = np.sum(demand_R == 0)
        print(f"  [住宅区 (R)] 共 {len(idx_R)} 个节点 | 出现零需求节点数: {zeros_R}")
        print("  需求明细: " + ", ".join([f"{val:.1f}" for val in demand_R]))

        # 打印商业区
        demand_C = single_snapshot[idx_C]
        zeros_C = np.sum(demand_C == 0)
        print(f"  [商业区 (C)] 共 {len(idx_C)} 个节点 | 出现零需求节点数: {zeros_C}")
        print("  需求明细: " + ", ".join([f"{val:.1f}" for val in demand_C]))

if __name__ == '__main__':
    inspect_single_snapshots()