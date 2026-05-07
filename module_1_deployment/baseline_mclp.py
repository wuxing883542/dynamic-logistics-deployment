import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 全局视觉设置
# ==========================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv

# ==========================================================
# 动态加载 RL 推理结果
# ==========================================================
def load_rl_best_hubs():
    rl_info_path = os.path.join(project_root, 'data', 'rl_inference_best.pkl')
    if os.path.exists(rl_info_path):
        with open(rl_info_path, 'rb') as f:
            best_info = pickle.load(f)
            print(f"✅ 成功加载动态推理结果: {best_info['best_hubs']}")
            return best_info['best_hubs']
    else:
        print("⚠️ 未发现推理结果文件，降级使用默认方案 [2, 5, 18]")
        return [2, 5, 18]

# ==========================================================
# 对偶上升法路由分配 (Dual Ascent Routing) - 纯物理裁判引擎
# ==========================================================
def get_dual_ascent_routing(env, active_hubs, snap_demand, max_radius=None):
    if not active_hubs:
        return np.zeros((env.N, 1)), 0, 0
    num_hubs = len(active_hubs)
    C_dist = np.zeros((env.N, num_hubs + 1))
    
    for j, h_idx in enumerate(active_hubs):
        dists = env.dist_matrix[:, h_idx]
        costs = env.physics.calculate_economic_cost(distance=dists, demand=1.0)
        
        # 👑 绝对物理封锁：限制评估时的无人机最大飞行距离
        if max_radius is not None:
            # 超出设定半径的节点，运费设为天价，逼迫系统直接拒单！
            costs[dists > max_radius] = 1e8 
            
        C_dist[:, j] = costs
        
    C_dist[:, -1] = env.cfg.penalty_unmet 

    shadow_prices = np.zeros(num_hubs + 1)
    
    # 影子价格迭代寻优 (加入早停防震荡机制)
    for _ in range(20):
        previous_shadow = shadow_prices.copy()
        
        perceived_cost = C_dist + shadow_prices
        min_cost = np.min(perceived_cost, axis=1, keepdims=True)
        exp_cost = np.exp(-(perceived_cost - min_cost) / env.sinkhorn_temp)
        probs = exp_cost / np.sum(exp_cost, axis=1, keepdims=True)
        
        hub_loads = np.zeros(num_hubs)
        for j, h_idx in enumerate(active_hubs):
            mask = np.ones(env.N, dtype=bool); mask[h_idx] = False
            hub_loads[j] = np.sum(probs[mask, j] * snap_demand[mask])
            
        overloads = np.maximum(0, hub_loads - env.cfg.Q)
        shadow_prices[:-1] += overloads * env.shadow_step
        
        shadow_change = shadow_prices - previous_shadow
        if np.max(np.abs(shadow_change)) < 1e-3:
            break 

    # 硬分配逻辑
    hard_probs = np.zeros_like(probs)
    remaining_capacity = np.ones(num_hubs) * env.cfg.Q
    node_order = np.argsort(-snap_demand) 

    for i in node_order:
        req = snap_demand[i]
        if probs[i, -1] > 0.5:
            hard_probs[i, -1] = 1.0; continue
            
        preferred_hubs = np.argsort(-probs[i, :-1])
        assigned = False
        for j in preferred_hubs:
            hub_node = active_hubs[j]
            if i == hub_node: 
                hard_probs[i, j] = 1.0; assigned = True; break
            elif remaining_capacity[j] >= req:
                hard_probs[i, j] = 1.0; remaining_capacity[j] -= req; assigned = True; break
        if not assigned: hard_probs[i, -1] = 1.0 

    transport_cost = 0.0; idle_penalty = 0.0; actual_hub_loads = np.zeros(num_hubs)
    for j, h_idx in enumerate(active_hubs):
        mask = np.ones(env.N, dtype=bool); mask[h_idx] = False 
        transport_cost += np.sum(hard_probs[mask, j] * C_dist[mask, j] * snap_demand[mask])
        actual_hub_loads[j] = np.sum(hard_probs[mask, j] * snap_demand[mask])
        
        hub_idle = max(0, env.cfg.Q - actual_hub_loads[j])
        idle_penalty += hub_idle * (env.cfg.penalty_unmet * 0.005)

    penalty_cost = np.sum(hard_probs[:, -1] * snap_demand) * env.cfg.penalty_unmet
    soft_overloads = np.maximum(0, actual_hub_loads - env.cfg.Q)
    
    overload_penalty = np.sum(soft_overloads) * env.overload_coef
    
    return hard_probs, transport_cost + penalty_cost + overload_penalty, idle_penalty

# ==========================================================
# 容量感知型 MCLP (Capacity-Aware MCLP)
# ==========================================================
def get_capacity_aware_mclp_hubs(env, radius_m, max_hubs):
    M = env.cfg.M_snapshots
    peak_indices = [M//2 + t*M for t in range(env.cfg.T_periods)]
    candidate_votes = np.zeros(env.N)
    
    for idx in peak_indices:
        demand = env.snapshots[idx]
        uncovered = set(range(env.N))
        local_hubs = []
        
        while uncovered and len(local_hubs) < max_hubs:
            best_j, max_cov = -1, -1
            for j in range(env.N):
                if j in local_hubs: continue
                
                simulated_capacity = env.cfg.Q
                cov = 0
                nodes_in_range = [i for i in uncovered if env.dist_matrix[i, j] <= radius_m]
                nodes_in_range.sort(key=lambda x: demand[x], reverse=True)
                
                for i in nodes_in_range:
                    req = demand[i]
                    if i == j: 
                        cov += req
                    elif simulated_capacity >= req:
                        cov += req
                        simulated_capacity -= req
                    else:
                        break 
                        
                if cov > max_cov:
                    max_cov = cov
                    best_j = j
                    
            if best_j != -1 and max_cov > 0:
                local_hubs.append(best_j)
                
                simulated_capacity = env.cfg.Q
                nodes_in_range = [i for i in uncovered if env.dist_matrix[i, best_j] <= radius_m]
                nodes_in_range.sort(key=lambda x: demand[x], reverse=True)
                for i in nodes_in_range:
                    req = demand[i]
                    if i == best_j:
                        uncovered.remove(i)
                    elif simulated_capacity >= req:
                        simulated_capacity -= req
                        uncovered.remove(i)
                    else:
                        break
            else: 
                break
                
        for h in local_hubs: 
            candidate_votes[h] += 1
            
    return np.argsort(-candidate_votes)[:max_hubs].tolist()

# ==========================================================
# 全量快照评估 
# ==========================================================
def evaluate_hubs(env, hubs, max_radius=None):
    op_costs = []
    for demand in env.snapshots:
        _, op_cost, idle_penalty = get_dual_ascent_routing(env, hubs, demand, max_radius)
        op_costs.append(op_cost + idle_penalty)
    return np.mean(op_costs), np.std(op_costs)

# ==========================================================
# 主流程与图表绘制
# ==========================================================
def run_comparison():
    cfg = UAVHubConfig(); env = RobustHubEnv(cfg)
    output_dir = os.path.join(project_root, 'data', 'MCLP_Baseline_Results')
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Ours: RL 评估
    # ----------------------------------------------------
    rl_hubs = load_rl_best_hubs()
    rl_cost, rl_std = evaluate_hubs(env, rl_hubs, max_radius=None)
    print(f"\n🚀 [Ours] RL 纯数据驱动: {rl_hubs} | 纯运营成本: {rl_cost:.0f} | 波动率: {rl_std:.0f}")

    # ----------------------------------------------------
    # Baseline: 容量感知型 MCLP 扫描 
    # ----------------------------------------------------
    radii = [150, 180, 200, 220, 250, 280, 300, 350, 400, 500]
    mclp_results = []
    print(f"📉 [Baseline] 启动同等规模 (强制 {len(rl_hubs)} 枢纽) 且带物理封锁的 MCLP 扫描...")
    for r in radii:
        hubs = get_capacity_aware_mclp_hubs(env, r, max_hubs=len(rl_hubs))
        cost, std = evaluate_hubs(env, hubs, max_radius=r) 
        mclp_results.append({'r': r, 'hubs': hubs, 'cost': cost, 'std': std})
        print(f"   ➤ 半径 {r:3d}m -> 选址 {str(hubs):<15} | 运营成本: {cost:8.0f} | 波动: {std:7.0f}")

    best_mclp_cost_idx = np.argmin([res['cost'] for res in mclp_results])
    best_mclp_std_idx = np.argmin([res['std'] for res in mclp_results])
    
    best_cost_res = mclp_results[best_mclp_cost_idx]
    best_std_res = mclp_results[best_mclp_std_idx]

    # ==================== 图 1: U 型运营成本图 ====================
    plt.figure(figsize=(10, 6))
    r_values = [res['r'] for res in mclp_results]
    mclp_costs = [res['cost'] for res in mclp_results]
    
    plt.plot(r_values, mclp_costs, marker='o', linewidth=2.5, color='#E74C3C', label='MCLP 启发式基准')
    plt.axhline(y=rl_cost, color='#2ECC71', linestyle='--', linewidth=3, label=f'Ours: RL 纯数据驱动决策 ({len(rl_hubs)}枢纽)')
    
    plt.annotate(f"MCLP 下限 ({best_cost_res['r']}m)\n{best_cost_res['cost']:.0f}", 
                 xy=(best_cost_res['r'], best_cost_res['cost']), 
                 xytext=(0, 40), textcoords="offset points", 
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', fontsize=10, fontweight='bold', color='#E74C3C')

    plt.title("期望运营成本对比：同等资产规模下的极限调度博弈", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("MCLP 人工设定的服务半径 (米)", fontsize=12)
    plt.ylabel("期望运营与惩罚成本 (剔除常数项建站费)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    
    y_ceiling = max(rl_cost, best_cost_res['cost']) * 2.5
    plt.ylim(rl_cost * 0.85, y_ceiling) 

    plt.savefig(os.path.join(output_dir, "01_MCLP_U_Shape_Cost.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 图 2: 鲁棒性双柱图 ====================
    # 👑 听你的，直接用最强基准 1v1 单挑
    plt.figure(figsize=(8, 6))
    labels = [
        f"MCLP (最优鲁棒基准)\nr={best_std_res['r']}m", 
        f"Ours: RL 纯数据驱动\n({len(rl_hubs)}枢纽)"
    ]
    stds = [best_std_res['std'], rl_std]
    colors = ['#E74C3C', '#2ECC71']
    
    bars = plt.bar(labels, stds, color=colors, width=0.4, edgecolor='black', linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(stds)*0.02), f'{yval:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.title("同等规模下的极限抗压能力 (全天运营成本波动率)", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("运营成本标准差 (越低越稳)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 加一点顶部留白，让数字不被切掉
    plt.ylim(0, max(stds) * 1.15)
    
    plt.savefig(os.path.join(output_dir, "02_Robustness_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 图 3: Pareto 散点图 ====================
    plt.figure(figsize=(9, 7))
    mclp_stds_all = [res['std'] for res in mclp_results]
    
    plt.scatter(mclp_costs, mclp_stds_all, c='red', s=100, label='MCLP 参数扫描方案集合', alpha=0.6, edgecolors='black')
    for res in mclp_results:
        plt.annotate(f"{res['r']}m", (res['cost'], res['std']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.scatter([rl_cost], [rl_std], c='#2ECC71', s=300, marker='*', label=f'Ours: RL 策略 ({len(rl_hubs)}枢纽)', edgecolors='black', zorder=5)
    plt.title("决策空间分析: 纯运营成本 vs 潮汐鲁棒性 (Pareto 面)", fontsize=14, fontweight='bold')
    plt.xlabel("期望运营与惩罚成本 (Lower is better)", fontsize=12)
    plt.ylabel("运营成本标准差 (Lower is more robust)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.xlim(rl_cost * 0.95, max(rl_cost, best_cost_res['cost']) * 2.5)
    plt.ylim(rl_std * 0.5, best_std_res['std'] * 2.5)

    plt.savefig(os.path.join(output_dir, "03_Pareto_Frontier.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 双柱对比图已生成！图表已更新至: {output_dir}")

if __name__ == "__main__":
    run_comparison()