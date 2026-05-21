"""
分时覆盖率曲线对比：RL vs Greedy
一张图讲清楚时序公平性 —— 这是项目最核心的实验证据。
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，仅保存图片
import matplotlib.pyplot as plt
import torch
from collections import deque
from scipy.interpolate import make_interp_spline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import DynamicDispatchPPO, FutureDemandPredictor


def run_rl_temporal(cfg, env, ppo, predictor, device, scenarios):
    """RL 策略：记录每步局部覆盖率"""
    N = env.N
    T = cfg.T_timesteps
    ORDER_SCALE = 100.0

    all_step_covs = np.zeros((len(scenarios), T))

    for sid, scenario in enumerate(scenarios):
        if sid % 5 == 0:
            print(f"  场景 {sid+1}/{len(scenarios)}...")
        # 重置环境到指定场景
        obs, _ = env.reset()
        env.current_scenario = scenario
        env.current_t = 0
        env.hub_capacities = np.full(env.K, cfg.Q, dtype=np.float32)
        env.ep_total_demand = 0.0
        env.ep_total_unmet = 0.0
        env.ep_step_covs = []
        env.ema_cov = None
        obs = env._get_obs()

        initial_orders = obs['current_orders'].copy()
        history_buffer = deque([initial_orders.copy() for _ in range(12)], maxlen=12)

        for t in range(T):
            # Predictor：先推入当前需求，再取历史窗口 (与训练逻辑严格一致)
            history_buffer.append(obs['current_orders'].copy())
            hist_arr = np.array(history_buffer, dtype=np.float32)
            hist_tensor = torch.tensor(hist_arr, device=device).unsqueeze(0) / ORDER_SCALE
            pred_out = predictor(hist_tensor)  # (1, pred_len, N)
            predicted = pred_out[0, 0, :] * ORDER_SCALE

            # Policy
            o_node = torch.tensor(obs['node_features'], dtype=torch.float32, device=device)
            o_orders = torch.tensor(obs['current_orders'], dtype=torch.float32, device=device)
            o_mask = torch.tensor(obs['hub_mask'], dtype=torch.float32, device=device)
            o_cap = torch.tensor(obs['hub_capacities'], dtype=torch.float32, device=device)
            o_time = torch.tensor(obs['time_ratio'], dtype=torch.float32, device=device)
            a_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)

            step_obs = {
                'node_features': o_node,
                'current_orders': o_orders,
                'hub_mask': o_mask,
                'hub_capacities': o_cap,
                'predicted_orders': predicted,
                'time_ratio': o_time,
            }

            with torch.no_grad():
                action, _, _, _ = ppo.get_action(step_obs, action_mask=a_mask, deterministic=True)

            next_obs, _, terminated, _, info = env.step(action.cpu().numpy())
            all_step_covs[sid, t] = info['step_coverage']
            obs = next_obs
            if terminated:
                break

    # 每步跨场景取平均
    mean_covs = all_step_covs.mean(axis=0)
    std_covs = all_step_covs.std(axis=0)
    return mean_covs, std_covs


def run_greedy_temporal(cfg, env, scenarios):
    """贪心策略：记录每步局部覆盖率"""
    from sklearn.cluster import KMeans

    K = cfg.max_hubs
    T = cfg.T_timesteps
    Q = cfg.Q
    max_r = cfg.max_flight_radius
    penalty = cfg.penalty_unmet

    # 复现贪心所需的静态结构
    coords = env.coords
    base_intensity = env.base_intensity
    dist_matrix = env.dist_matrix
    N = env.N
    hubs = env.hub_locations

    mask = np.ones((N, K), dtype=bool)
    for i in range(N):
        for k in range(K):
            if dist_matrix[i, hubs[k]] > max_r:
                mask[i, k] = False

    all_step_covs = np.zeros((len(scenarios), T))

    for sid, scenario in enumerate(scenarios):
        rem_cap = np.full(K, Q, dtype=np.float64)
        for t in range(T):
            demand = scenario[t].copy()
            step_demand = float(demand.sum())
            step_allocated = 0.0

            active = np.where(demand > 0)[0]
            for i in active[np.argsort(-demand[active])]:
                d = demand[i]
                best_k, best_d = K, np.inf
                for k in range(K):
                    if mask[i, k] and rem_cap[k] > 1e-6:
                        dd = dist_matrix[i, hubs[k]]
                        if dd < best_d:
                            best_d, best_k = dd, k
                if best_k < K:
                    a = min(d, rem_cap[best_k])
                    step_allocated += a
                    rem_cap[best_k] -= a

            all_step_covs[sid, t] = step_allocated / max(step_demand, 1e-5)

    mean_covs = all_step_covs.mean(axis=0)
    std_covs = all_step_covs.std(axis=0)
    return mean_covs, std_covs


def main():
    cfg = UAVHubConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 加载数据 ──
    data_path = os.path.join(BASE_DIR, 'data', 'map_adaptive_seed42_robust.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    eval_scenarios = data['eval_scenarios']
    N = data['config_meta']['N']
    T = cfg.T_timesteps

    # ── 创建环境 ──
    env = RobustHubEnv(cfg, mode='eval')
    env.N = N  # 确保一致

    # ── 加载模型 (自动配对 policy + predictor) ──
    model_dir = os.path.join(BASE_DIR, 'module_1_deployment', 'models')
    policy_files = [f for f in os.listdir(model_dir) if f.startswith('ppo_policy_ep')]
    pred_files = set(f for f in os.listdir(model_dir) if f.startswith('predictor_ep'))
    # 找同时有 policy 和 predictor 的最大 episode
    paired_ep = -1
    for pf in policy_files:
        ep = int(pf.replace('ppo_policy_ep', '').replace('.pth', ''))
        pred_name = f'predictor_ep{ep}.pth'
        if pred_name in pred_files and ep > paired_ep:
            paired_ep = ep
    if paired_ep < 0:
        raise FileNotFoundError("未找到配对的 policy + predictor 模型，请等训练多存几个 checkpoint")
    print(f"加载模型: ppo_policy_ep{paired_ep}.pth + predictor_ep{paired_ep}.pth")

    predictor = FutureDemandPredictor(N=N, history_len=12, pred_len=4, hidden_dim=64).to(device)
    ppo = DynamicDispatchPPO(cfg, N=N, node_feature_dim=9, hidden_dim=128).to(device)

    predictor.load_state_dict(
        torch.load(os.path.join(model_dir, f'predictor_ep{paired_ep}.pth'),
                   map_location=device))
    ppo.load_state_dict(torch.load(os.path.join(model_dir, f'ppo_policy_ep{paired_ep}.pth'), map_location=device))

    predictor.eval()
    ppo.eval()

    # ── 运行对比 ──
    print("运行 RL 策略...")
    rl_mean, rl_std = run_rl_temporal(cfg, env, ppo, predictor, device, eval_scenarios)
    print(f"  RL: 平均步覆盖={rl_mean.mean():.3f}, 最差步={rl_mean.min():.3f}, Std={rl_mean.std():.3f}")

    print("运行贪心策略...")
    gr_mean, gr_std = run_greedy_temporal(cfg, env, eval_scenarios)
    print(f"  贪心: 平均步覆盖={gr_mean.mean():.3f}, 最差步={gr_mean.min():.3f}, Std={gr_mean.std():.3f}")

    # ── 绘图 ──
    plt.rcParams.update({'font.size': 13, 'figure.dpi': 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    hours = np.arange(T) * 24.0 / T
    smooth_T = 192  # 2x 平滑
    hours_smooth = np.linspace(0, 24, smooth_T)

    rl_smooth = make_interp_spline(hours, rl_mean, k=3)(hours_smooth)
    gr_smooth = make_interp_spline(hours, gr_mean, k=3)(hours_smooth)

    # 左图：覆盖曲线
    ax1.plot(hours_smooth, rl_smooth * 100, 'b-', linewidth=2.2, label='RL (Ours)')
    ax1.fill_between(hours, rl_mean * 100 - rl_std * 100, rl_mean * 100 + rl_std * 100,
                     alpha=0.12, color='b')
    ax1.plot(hours_smooth, gr_smooth * 100, 'r--', linewidth=2.2, label='Greedy')
    ax1.fill_between(hours, gr_mean * 100 - gr_std * 100, gr_mean * 100 + gr_std * 100,
                     alpha=0.12, color='r')

    ax1.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='70% floor')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Step Coverage (%)')
    ax1.set_title('Per-Step Coverage: RL vs Greedy')
    ax1.legend(loc='lower left')
    ax1.set_ylim(50, 100)
    ax1.set_xlim(0, 24)
    ax1.grid(True, alpha=0.3)

    # 右图：公平性指标柱状图
    metrics = ['Min Step\nCoverage', 'Std of Step\nCoverage', 'Episode\nCoverage']
    rl_vals = [rl_mean.min() * 100, rl_mean.std() * 100, rl_mean.mean() * 100]
    gr_vals = [gr_mean.min() * 100, gr_mean.std() * 100, gr_mean.mean() * 100]

    x = np.arange(len(metrics))
    width = 0.32
    ax2.bar(x - width / 2, rl_vals, width, color='b', alpha=0.85, label='RL (Ours)')
    ax2.bar(x + width / 2, gr_vals, width, color='r', alpha=0.55, label='Greedy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Temporal Fairness Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for i in range(len(metrics)):
        ax2.text(x[i] - width / 2, rl_vals[i] + 0.5, f'{rl_vals[i]:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='b')
        ax2.text(x[i] + width / 2, gr_vals[i] + 0.5, f'{gr_vals[i]:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='r')

    fig.suptitle('Temporal Coverage Fairness: RL vs Greedy Baseline',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, 'module_1_deployment', 'logs', 'temporal_fairness.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")


if __name__ == '__main__':
    main()
