"""
独立基准测试脚本：Myopic LP / MPC 滚动优化 / Clairvoyant LP + 贪心
不依赖项目任何模块（仅读取 data pickle）。
用于定位 RL 训练效果的理论上下界。
"""

import os
import sys
import pickle
import time
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from dataclasses import dataclass, field


# ============================================================
# 独立配置（与 config.py 参数一致）
# ============================================================
@dataclass
class BConfig:
    seed: int = 42
    Q: float = 9000.0
    max_flight_radius: float = 850.0
    max_hubs: int = 3
    map_size: float = 2000.0
    T_timesteps: int = 96
    penalty_unmet: float = 100.0


# ============================================================
# 数据加载与初始化
# ============================================================
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'map_adaptive_seed42_robust.pkl')
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def generate_hubs(coords, base_intensity, K, seed):
    """复现环境的 K-Means + 峰值贪婪枢纽选择"""
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(coords)
    hubs = []
    for k in range(K):
        cluster_idx = np.where(labels == k)[0]
        best = cluster_idx[np.argmax(base_intensity[cluster_idx])]
        hubs.append(int(best))
    return hubs


def build_spatial_mask(N, K, hubs, dist_matrix, max_radius):
    """mask[i, k] = True 表示节点 i 在枢纽 k 的服务半径内"""
    mask = np.ones((N, K), dtype=bool)
    for i in range(N):
        for k in range(K):
            if dist_matrix[i, hubs[k]] > max_radius:
                mask[i, k] = False
    return mask


# ============================================================
# LP 求解器
# ============================================================

def solve_myopic_step(demand, hubs, dist_matrix, spatial_mask, rem_cap, penalty_unmet):
    """
    单步最优运输 LP。
    min  Σ_i Σ_k (dist[i,k]*0.01 * x[i,k]) + Σ_i (penalty * u[i])
    s.t. Σ_k x[i,k] + u[i] = demand[i]      ∀i
         Σ_i x[i,k] ≤ rem_cap[k]            ∀k
         x[i,k]=0 if !spatial_mask[i,k], x,u ≥ 0
    返回: alloc (N,K), unmet (N,), success
    """
    N, K = len(demand), len(rem_cap)
    n_vars = N * (K + 1)

    # 目标系数
    c = np.zeros(n_vars)
    for i in range(N):
        for k in range(K):
            if spatial_mask[i, k]:
                c[i * (K + 1) + k] = dist_matrix[i, hubs[k]] * 0.01
        c[i * (K + 1) + K] = penalty_unmet

    # 等式约束：每个节点的需求必须被满足或拒单
    A_eq = np.zeros((N, n_vars))
    b_eq = demand.astype(np.float64)
    for i in range(N):
        offset = i * (K + 1)
        A_eq[i, offset:offset + K + 1] = 1.0

    # 不等式约束：每个枢纽的累积分配不超过剩余容量
    A_ub = np.zeros((K, n_vars))
    b_ub = rem_cap.astype(np.float64)
    for k in range(K):
        for i in range(N):
            A_ub[k, i * (K + 1) + k] = 1.0

    # 变量边界（空间掩码）
    bounds = [(0, None)] * n_vars
    for i in range(N):
        for k in range(K):
            if not spatial_mask[i, k]:
                bounds[i * (K + 1) + k] = (0, 0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return None, None, False

    x = res.x.reshape(N, K + 1)
    return x[:, :K], x[:, K], True


def solve_mpc_window(demands_window, hubs, dist_matrix, spatial_mask, rem_cap, penalty_unmet):
    """
    MPC H-step 前瞻 LP。
    demands_window: (H, N)，当前步 + 未来 H-1 步的真实需求。
    返回: alloc_t (N,K), unmet_t (N,), success  (仅执行第一个步)
    """
    H, N = demands_window.shape
    K = len(rem_cap)
    n_vars = H * N * (K + 1)

    c = np.zeros(n_vars)
    for tau in range(H):
        offset = tau * N * (K + 1)
        for i in range(N):
            for k in range(K):
                if spatial_mask[i, k]:
                    c[offset + i * (K + 1) + k] = dist_matrix[i, hubs[k]] * 0.01
            c[offset + i * (K + 1) + K] = penalty_unmet

    # 等式约束：每个时间步每个节点的需求
    A_eq = np.zeros((H * N, n_vars))
    b_eq = demands_window.flatten().astype(np.float64)
    for tau in range(H):
        for i in range(N):
            row = tau * N + i
            offset = tau * N * (K + 1) + i * (K + 1)
            A_eq[row, offset:offset + K + 1] = 1.0

    # 不等式约束：每个枢纽在窗内总分配不超过剩余容量
    A_ub = np.zeros((K, n_vars))
    b_ub = rem_cap.astype(np.float64)
    for k in range(K):
        for tau in range(H):
            offset = tau * N * (K + 1)
            for i in range(N):
                A_ub[k, offset + i * (K + 1) + k] = 1.0

    # 变量边界
    bounds = [(0, None)] * n_vars
    for tau in range(H):
        offset = tau * N * (K + 1)
        for i in range(N):
            for k in range(K):
                if not spatial_mask[i, k]:
                    bounds[offset + i * (K + 1) + k] = (0, 0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return None, None, False

    x = res.x.reshape(H, N, K + 1)
    return x[0, :, :K], x[0, :, K], True


def solve_clairvoyant(all_demands, hubs, dist_matrix, spatial_mask, total_cap, penalty_unmet):
    """
    全知 LP：压缩时间维度，一次性分配全天总需求。
    运输成本与时间无关 + 容量为全天累积 → 时间维度可消除。
    返回: coverage, alloc_total (N,K), success
    """
    T, N = all_demands.shape
    K = len(total_cap)
    total_demand = all_demands.sum(axis=0).astype(np.float64)
    n_vars = N * (K + 1)

    c = np.zeros(n_vars)
    for i in range(N):
        for k in range(K):
            if spatial_mask[i, k]:
                c[i * (K + 1) + k] = dist_matrix[i, hubs[k]] * 0.01
        c[i * (K + 1) + K] = penalty_unmet

    A_eq = np.zeros((N, n_vars))
    b_eq = total_demand
    for i in range(N):
        offset = i * (K + 1)
        A_eq[i, offset:offset + K + 1] = 1.0

    A_ub = np.zeros((K, n_vars))
    b_ub = total_cap.astype(np.float64)
    for k in range(K):
        for i in range(N):
            A_ub[k, i * (K + 1) + k] = 1.0

    bounds = [(0, None)] * n_vars
    for i in range(N):
        for k in range(K):
            if not spatial_mask[i, k]:
                bounds[i * (K + 1) + k] = (0, 0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        return None, None, False

    x = res.x.reshape(N, K + 1)
    total_served = x[:, :K].sum()
    coverage = total_served / max(total_demand.sum(), 1e-5)
    return coverage, x, True


# ============================================================
# 基准运行器
# ============================================================

def run_greedy(scenarios, hubs, dist_matrix, spatial_mask, Q, penalty_unmet, T, max_radius):
    """贪心：按需求降序，每节点选最近有容量枢纽。与之前验证的 68.4% 一致。"""
    N = dist_matrix.shape[0]
    K = len(hubs)
    coverages = []

    for sid, scenario in enumerate(scenarios):
        rem_cap = np.full(K, Q, dtype=np.float64)
        total_demand = 0.0
        total_served = 0.0

        for t in range(T):
            demand = scenario[t]
            total_demand += demand.sum()
            active = np.where(demand > 0)[0]
            for i in active[np.argsort(-demand[active])]:
                d = demand[i]
                best_k, best_d = K, np.inf
                for k in range(K):
                    if spatial_mask[i, k] and rem_cap[k] > 1e-6:
                        dist = dist_matrix[i, hubs[k]]
                        if dist < best_d:
                            best_d, best_k = dist, k
                if best_k < K:
                    alloc = min(d, rem_cap[best_k])
                    total_served += alloc
                    rem_cap[best_k] -= alloc

        coverages.append(total_served / max(total_demand, 1e-5))

    return np.mean(coverages), np.std(coverages)


def run_myopic_lp(scenarios, hubs, dist_matrix, spatial_mask, Q, penalty_unmet, T):
    """Myopic LP：每步独立求解最优运输问题。"""
    N = dist_matrix.shape[0]
    K = len(hubs)
    coverages = []
    fail_count = 0

    for scenario in scenarios:
        rem_cap = np.full(K, Q, dtype=np.float64)
        total_demand = 0.0
        total_served = 0.0
        ok = True

        for t in range(T):
            demand = scenario[t]
            total_demand += demand.sum()

            alloc, unmet, success = solve_myopic_step(
                demand, hubs, dist_matrix, spatial_mask, rem_cap, penalty_unmet
            )
            if not success:
                fail_count += 1
                ok = False
                break
            total_served += alloc.sum()
            rem_cap -= alloc.sum(axis=0)

        if ok:
            coverages.append(total_served / max(total_demand, 1e-5))

    if fail_count:
        print(f"    ⚠ Myopic LP 失败 {fail_count} 步")
    return np.mean(coverages) if coverages else 0.0, np.std(coverages) if coverages else 0.0


def run_mpc(scenarios, hubs, dist_matrix, spatial_mask, Q, penalty_unmet, T, H):
    """MPC：每步求解 H 步前瞻 LP，仅执行第一步。"""
    N = dist_matrix.shape[0]
    K = len(hubs)
    coverages = []
    fail_count = 0

    for scenario in scenarios:
        rem_cap = np.full(K, Q, dtype=np.float64)
        total_demand = 0.0
        total_served = 0.0
        ok = True

        for t in range(T):
            demand_t = scenario[t]
            total_demand += demand_t.sum()

            horizon = min(H, T - t)
            demands_window = scenario[t:t + horizon]

            alloc, unmet, success = solve_mpc_window(
                demands_window, hubs, dist_matrix, spatial_mask, rem_cap, penalty_unmet
            )
            if not success:
                fail_count += 1
                ok = False
                break
            total_served += alloc.sum()
            rem_cap -= alloc.sum(axis=0)

        if ok:
            coverages.append(total_served / max(total_demand, 1e-5))

    if fail_count:
        print(f"    ⚠ MPC(H={H}) 失败 {fail_count} 步")
    return np.mean(coverages) if coverages else 0.0, np.std(coverages) if coverages else 0.0


def run_clairvoyant(scenarios, hubs, dist_matrix, spatial_mask, Q, penalty_unmet, T):
    """Clairvoyant LP：全知分配全天总需求。"""
    K = len(hubs)
    total_cap = np.full(K, Q, dtype=np.float64)
    coverages = []

    for scenario in scenarios:
        cov, _, success = solve_clairvoyant(
            scenario, hubs, dist_matrix, spatial_mask, total_cap, penalty_unmet
        )
        if success:
            coverages.append(cov)

    return np.mean(coverages) if coverages else 0.0, np.std(coverages) if coverages else 0.0


# ============================================================
# 主入口
# ============================================================
def main():
    cfg = BConfig()

    print("=" * 70)
    print("📊 枢纽分配基准测试：贪心 / Myopic LP / MPC / Clairvoyant LP")
    print("=" * 70)

    data = load_data()
    topo = data['topo_data']
    train_scenarios = data['train_scenarios']
    eval_scenarios = data.get('eval_scenarios', train_scenarios[-30:])

    coords = topo['coords']
    base_intensity = topo['base_intensity']
    dist_matrix = topo['C']
    N = dist_matrix.shape[0]
    K = cfg.max_hubs
    T = cfg.T_timesteps

    hubs = generate_hubs(coords, base_intensity, K, cfg.seed)
    spatial_mask = build_spatial_mask(N, K, hubs, dist_matrix, cfg.max_flight_radius)

    print(f"\n节点数: {N}  枢纽数: {K}  时间步: {T}")
    print(f"容量/枢纽: {cfg.Q:.0f}  最大半径: {cfg.max_flight_radius:.0f}m")
    print(f"枢纽位置: {hubs}")
    print(f"训练场景: {len(train_scenarios)}  评估场景: {len(eval_scenarios)}")

    # ── 评估集基准 ──
    print("\n" + "=" * 70)
    print("🔬 评估集基准 (30 个场景)")
    print("=" * 70)

    results = {}

    t0 = time.time()
    mean, std = run_greedy(eval_scenarios, hubs, dist_matrix, spatial_mask,
                           cfg.Q, cfg.penalty_unmet, T, cfg.max_flight_radius)
    results['贪心 (Greedy)'] = (mean, std)
    print(f"  [1/4] 贪心          → {mean*100:.2f}% ± {std*100:.2f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    mean, std = run_myopic_lp(eval_scenarios, hubs, dist_matrix, spatial_mask,
                              cfg.Q, cfg.penalty_unmet, T)
    results['Myopic LP (H=1)'] = (mean, std)
    print(f"  [2/4] Myopic LP     → {mean*100:.2f}% ± {std*100:.2f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    mean, std = run_mpc(eval_scenarios, hubs, dist_matrix, spatial_mask,
                        cfg.Q, cfg.penalty_unmet, T, H=4)
    results['MPC (H=4)'] = (mean, std)
    print(f"  [3/4] MPC H=4       → {mean*100:.2f}% ± {std*100:.2f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    mean, std = run_mpc(eval_scenarios, hubs, dist_matrix, spatial_mask,
                        cfg.Q, cfg.penalty_unmet, T, H=8)
    results['MPC (H=8)'] = (mean, std)
    print(f"  [4/4] MPC H=8       → {mean*100:.2f}% ± {std*100:.2f}%  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    mean, std = run_clairvoyant(eval_scenarios, hubs, dist_matrix, spatial_mask,
                                cfg.Q, cfg.penalty_unmet, T)
    results['Clairvoyant LP'] = (mean, std)
    print(f"  [5/5] Clairvoyant   → {mean*100:.2f}% ± {std*100:.2f}%  ({time.time()-t0:.1f}s)")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("📈 评估集汇总")
    print("=" * 70)
    print(f"  {'方法':<22} {'覆盖率':>10} {'±std':>8}")
    print(f"  {'─'*40}")
    for name, (mean, std) in results.items():
        print(f"  {name:<22} {mean*100:>8.2f}% {std*100:>7.2f}%")
    print(f"  {'─'*40}")

    # ── 关键差距分析 ──
    greedy_cov = results['贪心 (Greedy)'][0]
    myopic_cov = results['Myopic LP (H=1)'][0]
    mpc4_cov = results['MPC (H=4)'][0]
    mpc8_cov = results['MPC (H=8)'][0]
    clair_cov = results['Clairvoyant LP'][0]

    print(f"\n📋 差距链 (评估集):")
    print(f"  贪心 {greedy_cov*100:.1f}% "
          f"→ Myopic {myopic_cov*100:.1f}% (Δ={-(greedy_cov-myopic_cov)*100:+.1f}%)"
          f"→ MPC4 {mpc4_cov*100:.1f}% (Δ={-(myopic_cov-mpc4_cov)*100:+.1f}%)"
          f"→ MPC8 {mpc8_cov*100:.1f}% (Δ={-(mpc4_cov-mpc8_cov)*100:+.1f}%)"
          f"→ Clairvoyant {clair_cov*100:.1f}% (Δ={-(mpc8_cov-clair_cov)*100:+.1f}%)")

    print(f"\n  💡 预测价值 (Myopic→MPC4): {(mpc4_cov-myopic_cov)*100:+.1f}%")
    print(f"  💡 预测价值 (MPC4→MPC8):   {(mpc8_cov-mpc4_cov)*100:+.1f}%")
    print(f"  💡 不可达空间 (MPC8→上帝): {(clair_cov-mpc8_cov)*100:+.1f}%")
    print(f"  💡 理论天花板:             {clair_cov*100:.1f}%")

    print("\n✅ 基准测试完成。")


if __name__ == '__main__':
    main()
