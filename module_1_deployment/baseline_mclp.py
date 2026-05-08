import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
            print(f"成功加载 RL 推理结果: {best_info['best_hubs']}")
            return best_info['best_hubs']
    else:
        print("未发现推理结果文件，降级使用默认方案 [2, 5, 18]")
        return [2, 5, 18]

# ==========================================================
# 分配策略 1: Ours RL 专属 - 对偶上升智能路由 (联合优化)
# ==========================================================
def get_dual_ascent_routing(env, active_hubs, snap_demand, max_radius=None):
    if not active_hubs:
        return np.zeros((env.N, 1)), 0, 0, 0
    num_hubs = len(active_hubs)
    C_dist = np.zeros((env.N, num_hubs + 1))
    
    for j, h_idx in enumerate(active_hubs):
        dists = env.dist_matrix[:, h_idx]
        costs = env.physics.calculate_economic_cost(distance=dists, demand=1.0)
        if max_radius is not None:
            costs[dists > max_radius] = 1e8 
        C_dist[:, j] = costs
        
    C_dist[:, -1] = env.cfg.penalty_unmet 
    shadow_prices = np.zeros(num_hubs + 1)
    
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
        if np.max(np.abs(shadow_prices - previous_shadow)) < 1e-3:
            break 

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

    penalty_unmet = np.sum(hard_probs[:, -1] * snap_demand) * env.cfg.penalty_unmet
    soft_overloads = np.maximum(0, actual_hub_loads - env.cfg.Q)
    penalty_overload = np.sum(soft_overloads) * env.overload_coef
    
    return hard_probs, transport_cost, penalty_unmet + penalty_overload, idle_penalty

# ==========================================================
# 分配策略 2: K-Means/Random 原生专属 - 最近邻分配 (无视容量)
# ==========================================================
def get_kmeans_native_routing(env, active_hubs, snap_demand):
    if not active_hubs: return np.zeros((env.N, 1)), 0, 0, 0
    num_hubs = len(active_hubs)
    hard_probs = np.zeros((env.N, num_hubs + 1))
    
    # 原生逻辑：找最近的枢纽硬塞
    for i in range(env.N):
        dists = [env.dist_matrix[i, h] for h in active_hubs]
        best_j = np.argmin(dists)
        hard_probs[i, best_j] = 1.0

    transport_cost = 0.0; idle_penalty = 0.0; actual_hub_loads = np.zeros(num_hubs)
    for j, h_idx in enumerate(active_hubs):
        mask = np.ones(env.N, dtype=bool); mask[h_idx] = False 
        dists = env.dist_matrix[:, h_idx]
        costs = env.physics.calculate_economic_cost(distance=dists, demand=1.0)
        transport_cost += np.sum(hard_probs[mask, j] * costs[mask] * snap_demand[mask])
        actual_hub_loads[j] = np.sum(hard_probs[mask, j] * snap_demand[mask])
        hub_idle = max(0, env.cfg.Q - actual_hub_loads[j])
        idle_penalty += hub_idle * (env.cfg.penalty_unmet * 0.005)

    # 由于无视容量强塞，全部转化为超载罚款
    penalty_unmet = 0.0
    soft_overloads = np.maximum(0, actual_hub_loads - env.cfg.Q)
    penalty_overload = np.sum(soft_overloads) * env.overload_coef
    
    return hard_probs, transport_cost, penalty_unmet + penalty_overload, idle_penalty

# ==========================================================
# 分配策略 3: MCLP 原生专属 - 贪心边界分配 (严格卡死半径和容量)
# ==========================================================
def get_mclp_native_routing(env, active_hubs, snap_demand, max_radius):
    if not active_hubs: return np.zeros((env.N, 1)), 0, 0, 0
    num_hubs = len(active_hubs)
    hard_probs = np.zeros((env.N, num_hubs + 1))
    remaining_capacity = np.ones(num_hubs) * env.cfg.Q
    node_order = np.argsort(-snap_demand) 

    for i in node_order:
        req = snap_demand[i]
        # 只看半径内的枢纽
        valid_hubs = [(j, env.dist_matrix[i, active_hubs[j]]) for j in range(num_hubs) if env.dist_matrix[i, active_hubs[j]] <= max_radius]
        if not valid_hubs:
            hard_probs[i, -1] = 1.0; continue
            
        valid_hubs.sort(key=lambda x: x[1])
        assigned = False
        for j, dist in valid_hubs:
            if i == active_hubs[j]: 
                hard_probs[i, j] = 1.0; assigned = True; break
            elif remaining_capacity[j] >= req:
                hard_probs[i, j] = 1.0; remaining_capacity[j] -= req; assigned = True; break
        if not assigned: hard_probs[i, -1] = 1.0 

    transport_cost = 0.0; idle_penalty = 0.0; actual_hub_loads = np.zeros(num_hubs)
    for j, h_idx in enumerate(active_hubs):
        mask = np.ones(env.N, dtype=bool); mask[h_idx] = False 
        dists = env.dist_matrix[:, h_idx]
        costs = env.physics.calculate_economic_cost(distance=dists, demand=1.0)
        transport_cost += np.sum(hard_probs[mask, j] * costs[mask] * snap_demand[mask])
        actual_hub_loads[j] = np.sum(hard_probs[mask, j] * snap_demand[mask])
        hub_idle = max(0, env.cfg.Q - actual_hub_loads[j])
        idle_penalty += hub_idle * (env.cfg.penalty_unmet * 0.005)

    penalty_unmet = np.sum(hard_probs[:, -1] * snap_demand) * env.cfg.penalty_unmet
    soft_overloads = np.maximum(0, actual_hub_loads - env.cfg.Q)
    penalty_overload = np.sum(soft_overloads) * env.overload_coef
    
    return hard_probs, transport_cost, penalty_unmet + penalty_overload, idle_penalty

# ==========================================================
# 基线选址算法 (回归解耦状态)
# ==========================================================
def get_mclp_hubs(env, radius_m, max_hubs):
    M = env.cfg.M_snapshots
    sample_indices = [M//2, M + M//2, 2*M + M//2, 3*M + M//2]
    candidate_votes = np.zeros(env.N)

    for idx in sample_indices:
        demand = env.snapshots[idx]
        uncovered = set(range(env.N))
        local_hubs = []
        while uncovered and len(local_hubs) < max_hubs:
            best_j, max_cov = -1, -1
            for j in range(env.N):
                if j in local_hubs: continue
                simulated_cap = env.cfg.Q; cov = 0
                nodes_in_range = [i for i in uncovered if env.dist_matrix[i, j] <= radius_m]
                nodes_in_range.sort(key=lambda x: demand[x], reverse=True)
                for i in nodes_in_range:
                    req = demand[i]
                    if i == j or simulated_cap >= req:
                        cov += req
                        if i != j: simulated_cap -= req
                    else: break 
                if cov > max_cov:
                    max_cov = cov; best_j = j
            if best_j != -1 and max_cov > 0:
                local_hubs.append(best_j)
                simulated_cap = env.cfg.Q
                nodes_in_range = [i for i in uncovered if env.dist_matrix[i, best_j] <= radius_m]
                nodes_in_range.sort(key=lambda x: demand[x], reverse=True)
                for i in nodes_in_range:
                    req = demand[i]
                    if i == best_j: uncovered.remove(i)
                    elif simulated_cap >= req:
                        simulated_cap -= req; uncovered.remove(i)
                    else: break
            else: break
        for h in local_hubs: candidate_votes[h] += 1
    return np.argsort(-candidate_votes)[:max_hubs].tolist()

def get_kmeans_hubs(env, max_hubs):
    kmeans = KMeans(n_clusters=max_hubs, random_state=42, n_init=10)
    kmeans.fit(env.coords) 
    centers = kmeans.cluster_centers_
    hubs = []
    for center in centers:
        dists = np.linalg.norm(env.coords - center, axis=1)
        sorted_indices = np.argsort(dists)
        for idx in sorted_indices:
            if idx not in hubs:
                hubs.append(idx)
                break
    return hubs

# ==========================================================
# 统一评估入口 (按各自派系采用不同的原生分配机制)
# ==========================================================
def evaluate_hubs(env, hubs, strategy='rl', max_radius=None):
    fixed_c = sum([env.fixed_costs[h] for h in hubs])
    trans_list, pen_list, idle_list, total_list = [], [], [], []
    
    for demand in env.snapshots:
        # 🔥 核心：解耦算法用原生分配，RL 用联合智能分配
        if strategy == 'rl':
            _, t_cost, p_cost, i_cost = get_dual_ascent_routing(env, hubs, demand, max_radius)
        elif strategy == 'kmeans':
            _, t_cost, p_cost, i_cost = get_kmeans_native_routing(env, hubs, demand)
        elif strategy == 'mclp':
            _, t_cost, p_cost, i_cost = get_mclp_native_routing(env, hubs, demand, max_radius)

        total_op = t_cost + p_cost + i_cost
        trans_list.append(t_cost)
        pen_list.append(p_cost)
        idle_list.append(i_cost)
        total_list.append(fixed_c + total_op)
        
    M = env.cfg.M_snapshots
    T = env.cfg.T_periods
    period_costs = []
    for t in range(T):
        p_costs = total_list[t*M : (t+1)*M]
        period_costs.append(np.mean(p_costs))
        
    return {
        'total_mean': np.mean(total_list),
        'total_std': np.std(total_list),
        'fixed': fixed_c,
        'trans': np.mean(trans_list),
        'penalty': np.mean(pen_list),
        'idle': np.mean(idle_list),
        'period_costs': period_costs
    }

def get_random_baseline(env, max_hubs, num_trials=30):
    np.random.seed(42)
    all_res = []
    for _ in range(num_trials):
        hubs = list(np.random.choice(env.N, max_hubs, replace=False))
        # 随机基准采用与 K-Means 相同的原生最近邻分配
        all_res.append(evaluate_hubs(env, hubs, strategy='kmeans'))
        
    avg_period = np.mean([res['period_costs'] for res in all_res], axis=0).tolist()
    return {
        'total_mean': np.mean([r['total_mean'] for r in all_res]),
        'total_std': np.mean([r['total_std'] for r in all_res]),
        'fixed': all_res[0]['fixed'],
        'trans': np.mean([r['trans'] for r in all_res]),
        'penalty': np.mean([r['penalty'] for r in all_res]),
        'idle': np.mean([r['idle'] for r in all_res]),
        'period_costs': avg_period
    }

# ==========================================================
# 主流程与图表绘制
# ==========================================================
def run_comparison():
    cfg = UAVHubConfig(); env = RobustHubEnv(cfg)
    output_dir = os.path.join(project_root, 'data', 'MCLP_Baseline_Results')
    os.makedirs(output_dir, exist_ok=True)
    
    hub_count = len(load_rl_best_hubs())
    
    print("\n--- 启动策略评估 (区分联合优化与解耦模式) ---")
    print("评测基线: Random (解耦原生分配)")
    res_rand = get_random_baseline(env, hub_count)
    
    print("评测基线: K-Means (解耦原生分配)")
    km_hubs = get_kmeans_hubs(env, hub_count)
    res_km = evaluate_hubs(env, km_hubs, strategy='kmeans')
    
    print("评测策略: RL (端到端联合优化)")
    rl_hubs = load_rl_best_hubs()
    res_rl = evaluate_hubs(env, rl_hubs, strategy='rl')

    radii = [280, 300, 320, 340, 350, 360, 380, 400, 450, 500]
    mclp_results = []
    print("评测基线: MCLP 参数扫描 (解耦原生分配)")
    for r in radii:
        hubs = get_mclp_hubs(env, r, max_hubs=hub_count)
        res = evaluate_hubs(env, hubs, strategy='mclp', max_radius=r)
        res['r'] = r; res['hubs'] = hubs
        mclp_results.append(res)
        print(f"  半径 {r:3d}m | 总成本: {res['total_mean']:8.0f} | 波动: {res['total_std']:7.0f}")

    best_mclp = min(mclp_results, key=lambda x: x['total_mean'])

    # ==================== 图 1: MCLP 参数扫描曲线 ====================
    plt.figure(figsize=(10, 6))
    plt.plot([r['r'] for r in mclp_results], [r['total_mean'] for r in mclp_results], marker='o', linewidth=2.5, color='#E74C3C', label='MCLP 参数扫描')
    plt.axhline(y=res_rl['total_mean'], color='#2ECC71', linestyle='-', linewidth=3, label='Ours: RL 联合优化策略')
    plt.axhline(y=res_km['total_mean'], color='#3498DB', linestyle='-.', linewidth=2, label='K-Means 聚类分配')
    plt.axhline(y=res_rand['total_mean'], color='gray', linestyle='--', linewidth=2, label='Random 基准')
    
    plt.annotate(f"MCLP 极值 ({best_mclp['r']}m)\n{best_mclp['total_mean']:.0f}", 
                 xy=(best_mclp['r'], best_mclp['total_mean']), 
                 xytext=(0, 25), textcoords="offset points", 
                 arrowprops=dict(facecolor='#E74C3C', shrink=0.05, width=1.5, headwidth=6),
                 ha='center', fontsize=10, color='#C0392B')

    plt.title("期望运营成本对比：不同覆盖半径下的策略表现 (Q=200)", fontsize=14, pad=15)
    plt.xlabel("MCLP 服务半径约束 (米)", fontsize=12)
    plt.ylabel("全天期望综合成本", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    
    all_means = [r['total_mean'] for r in mclp_results] + [res_rl['total_mean'], res_km['total_mean'], res_rand['total_mean']]
    min_mean, max_mean = min(all_means), max(all_means)
    padding = (max_mean - min_mean) * 0.15
    plt.ylim(min_mean - padding, max_mean + padding) 
    plt.savefig(os.path.join(output_dir, "01_Parameter_Scan_Curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 图 2: 成本拆解堆叠 + 双轴鲁棒图 ====================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    labels = ['Random\n(解耦模式)', 'K-Means\n(解耦模式)', 'MCLP Best\n(解耦模式)', 'Ours: RL\n(联合优化)']
    data = [res_rand, res_km, best_mclp, res_rl]
    
    fixed_vals, trans_vals, pen_vals, idle_vals, std_vals = [d['fixed'] for d in data], [d['trans'] for d in data], [d['penalty'] for d in data], [d['idle'] for d in data], [d['total_std'] for d in data]
    x = np.arange(len(labels)); width = 0.45
    
    b1 = ax1.bar(x, fixed_vals, width, label='固定建站费', color='#BDC3C7', edgecolor='black')
    b2 = ax1.bar(x, trans_vals, width, bottom=fixed_vals, label='运营运费', color='#3498DB', edgecolor='black')
    b3 = ax1.bar(x, pen_vals, width, bottom=np.array(fixed_vals)+np.array(trans_vals), label='超载/拒单惩罚', color='#E74C3C', edgecolor='black')
    b4 = ax1.bar(x, idle_vals, width, bottom=np.array(fixed_vals)+np.array(trans_vals)+np.array(pen_vals), label='闲置成本', color='#F1C40F', edgecolor='black')
    
    min_fixed = min(fixed_vals)
    total_costs = [fixed_vals[i] + trans_vals[i] + pen_vals[i] + idle_vals[i] for i in range(4)]
    ax1.set_ylim(min_fixed * 0.95, max(total_costs) * 1.05)
    
    for i in range(len(labels)):
        ax1.text(x[i], total_costs[i] + (max(total_costs)*0.002), f"{total_costs[i]:.0f}", ha='center', va='bottom', fontsize=10, color='black')

    ax2.plot(x, std_vals, color='#2C3E50', marker='D', markersize=8, linewidth=2, linestyle='-', label='成本标准差 (Std)')
    for i, txt in enumerate(std_vals):
        ax2.annotate(f"{txt:.0f}", (x[i], std_vals[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#2C3E50')

    ax1.set_ylabel('全天期望综合成本', fontsize=12)
    ax2.set_ylabel('成本标准差 (右轴)', fontsize=12, color='#2C3E50')
    ax1.set_title(f"同等资产规模下的成本结构拆解与鲁棒性对比 (Q=200)", fontsize=14, pad=15)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, max(std_vals) * 1.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)
    plt.savefig(os.path.join(output_dir, "02_Cost_Decomposition_DualAxis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 图 3: Pareto 散点图 ====================
    plt.figure(figsize=(9, 6))
    plt.scatter([r['total_mean'] for r in mclp_results], [r['total_std'] for r in mclp_results], c='#E74C3C', s=80, label='MCLP 解耦组合', alpha=0.6, edgecolors='white')
    plt.scatter([res_rand['total_mean']], [res_rand['total_std']], c='#7F8C8D', s=120, marker='X', label='Random 基准', edgecolors='black')
    plt.scatter([res_km['total_mean']], [res_km['total_std']], c='#3498DB', s=120, marker='s', label='K-Means 解耦组合', edgecolors='black')
    plt.scatter([res_rl['total_mean']], [res_rl['total_std']], c='#2ECC71', s=250, marker='*', label='Ours: RL 联合优化', edgecolors='black', zorder=5)
    
    plt.title("决策空间分析：期望成本 vs 运营鲁棒性 (Q=200)", fontsize=14, pad=15)
    plt.xlabel("全天期望综合成本", fontsize=12)
    plt.ylabel("期望成本标准差", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    
    all_x = [r['total_mean'] for r in mclp_results] + [res_rand['total_mean'], res_km['total_mean'], res_rl['total_mean']]
    all_y = [r['total_std'] for r in mclp_results] + [res_rand['total_std'], res_km['total_std'], res_rl['total_std']]
    x_pad = (max(all_x) - min(all_x)) * 0.1
    y_pad = (max(all_y) - min(all_y)) * 0.1
    plt.xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    plt.ylim(min(all_y) - y_pad, max(all_y) + y_pad)
    plt.savefig(os.path.join(output_dir, "03_Pareto_Frontier.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ================================================================
    # 图 4: 容量压力测试
    # ================================================================
    print("\n--- 容量压力测试 (Q: 200 → 100) ---")
    Q_values = [200, 180, 160, 140, 120, 100]
    stress_random_hubs = list(np.random.RandomState(42).choice(env.N, hub_count, replace=False))
    stress_methods = {
        'Random':      (stress_random_hubs,      'kmeans', None,             'gray',    'X',  ':'),
        'K-Means':     (km_hubs,                 'kmeans', None,             '#3498DB', 's',  '--'),
        f'MCLP({best_mclp["r"]}m)': (best_mclp['hubs'], 'mclp',   best_mclp['r'], '#E74C3C', 'o',  '--'),
        'Ours: RL':    (rl_hubs,                 'rl',     None,             '#2ECC71', '*',  '-'),
    }

    stress_data = {name: [] for name in stress_methods}
    original_Q = env.cfg.Q

    for q in Q_values:
        env.cfg.Q = q
        env.overload_coef = env.cfg.penalty_unmet * 2.0
        for name, (hubs, strat, radius, _, _, _) in stress_methods.items():
            res = evaluate_hubs(env, hubs, strategy=strat, max_radius=radius)
            stress_data[name].append(res['total_mean'])

    env.cfg.Q = original_Q
    env.overload_coef = env.cfg.penalty_unmet * 2.0

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6))

    for name, (hubs, strat, radius, color, marker, ls) in stress_methods.items():
        ax4a.plot(Q_values, stress_data[name], marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=name)
    ax4a.set_title("全景视图 (对数轴)", fontsize=12)
    ax4a.set_xlabel("无人机载重约束 Q", fontsize=11)
    ax4a.set_ylabel("期望总成本 (对数)", fontsize=11)
    ax4a.set_yscale('log')
    ax4a.grid(True, linestyle='--', alpha=0.4)
    ax4a.legend(loc='upper left', fontsize=10)
    ax4a.invert_xaxis()

    competitive = ['Random', 'K-Means', 'Ours: RL']
    for name in competitive:
        _, _, _, color, marker, ls = stress_methods[name]
        ax4b.plot(Q_values, stress_data[name], marker=marker, color=color, linestyle=ls, linewidth=2, markersize=8, label=name)

    ax4b.set_title("常规约束区视图 (线性轴)", fontsize=12)
    ax4b.set_xlabel("无人机载重约束 Q", fontsize=11)
    ax4b.set_ylabel("期望总成本", fontsize=11)
    ax4b.grid(True, linestyle='--', alpha=0.4)
    ax4b.legend(loc='upper left', fontsize=10)
    ax4b.invert_xaxis()

    comp_all = stress_data['Random'] + stress_data['K-Means'] + stress_data['Ours: RL']
    y_pad_stress = (max(comp_all) - min(comp_all)) * 0.15
    ax4b.set_ylim(min(comp_all) - y_pad_stress, max(comp_all) + y_pad_stress)

    fig4.suptitle(f"容量压力测试：运力收缩对各策略综合成本的影响", fontsize=14)
    fig4.savefig(os.path.join(output_dir, "04_Capacity_Stress_Test.png"), dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # ================================================================
    # 图 5: 跨潮汐时段动态性能剖析 (常态 Q=200)
    # ================================================================
    print("\n--- 绘制时段切片分析图 ---")
    
    periods = ['早高峰\n(Morning)', '午平峰\n(Noon)', '晚高峰\n(Evening)', '深夜低谷\n(Night)']
    
    plt.figure(figsize=(9, 6))
    x_pos = np.arange(len(periods))
    
    plt.plot(x_pos, res_rand['period_costs'], marker='X', color='gray', linestyle=':', linewidth=2, markersize=8, label='Random 基准')
    plt.plot(x_pos, best_mclp['period_costs'], marker='o', color='#E74C3C', linestyle='--', linewidth=2, markersize=8, label='MCLP 解耦策略')
    plt.plot(x_pos, res_km['period_costs'], marker='s', color='#3498DB', linestyle='--', linewidth=2, markersize=8, label='K-Means 解耦策略')
    plt.plot(x_pos, res_rl['period_costs'], marker='*', color='#2ECC71', linestyle='-', linewidth=3, markersize=12, label='Ours: RL 联合优化')

    plt.title("典型潮汐时段内的平均综合成本演变分析 (Q=200)", fontsize=14, pad=15)
    plt.xlabel("全天典型潮汐时段", fontsize=12)
    plt.ylabel("时段平均综合成本", fontsize=12)
    plt.xticks(x_pos, periods, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)

    all_period_costs = res_rand['period_costs'] + best_mclp['period_costs'] + res_km['period_costs'] + res_rl['period_costs']
    plt.ylim(min(all_period_costs) * 0.98, max(all_period_costs) * 1.02)

    plt.savefig(os.path.join(output_dir, "05_Tidal_Period_Breakdown.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n评估图表生成完毕！保存在目录: {output_dir}")

if __name__ == "__main__":
    run_comparison()