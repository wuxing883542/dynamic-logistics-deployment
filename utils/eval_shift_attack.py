"""
突发峰值平移攻击测试 (Time-Shift Attack)
不修改底层数据集，仅在内存中将晚高峰平移至中午，测试策略的真实泛化能力。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import DynamicDispatchPPO, FutureDemandPredictor

SAVE_DIR = os.path.join(BASE_DIR, 'data', 'greedy_reserve_results')
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ORDER_SCALE = 100.0

def shift_scenarios_in_memory(env, shift_hours=-8):
    """
    💡 核心黑客科技：不改文件，只在内存里把数组滚动一下。
    shift_hours=-8 意味着把 18:00 的高峰提前 8 小时，挪到 10:00 左右。
    """
    shifted_scenarios = []
    for scenario in env.eval_scenarios:
        # scenario 形状是 (T, N)
        shifted = np.roll(scenario, shift=shift_hours, axis=0)
        shifted_scenarios.append(shifted)
    env.eval_scenarios = shifted_scenarios
    print(f"🌪️ [环境突变] 已成功将测试集在内存中平移 {shift_hours} 小时，高峰已转移至中午！")

def run_greedy_reserve(env, day_idx, reserve_ratio):
    """运行预留贪心策略"""
    env.current_scenario = env.eval_scenarios[day_idx]
    env.day_total_expected_demand = float(np.sum(env.current_scenario))
    env.current_t = 0
    env.hub_capacities = np.full(env.K, env.cfg.Q, dtype=np.float32)
    # 💡 补上这核心的两行，让环境知道今天的死目标是多少
    total_initial_cap = env.K * env.cfg.Q
    env.day_static_target = min(1.0, total_initial_cap / max(env.day_total_expected_demand, 1.0))


    obs = env._get_obs()
    done = False
    step_coverages = []

    while not done:
        action = np.full(env.N, env.K, dtype=np.int32)
        orders = obs['current_orders']
        active_nodes = np.where(orders > 0)[0]
        
        for i in active_nodes:
            valid_hubs = []
            for k in range(env.K):
                if env.dist_matrix[i, env.hub_locations[k]] <= env.max_radius:
                    valid_hubs.append(k)
            if valid_hubs:
                closest_hub = min(valid_hubs, key=lambda k: env.dist_matrix[i, env.hub_locations[k]])
                action[i] = closest_hub

        # 致命缺陷：死板的按时钟预留
        true_capacities = env.hub_capacities.copy()
        t_ratio = env.current_t / env.T
        frozen_cap = env.cfg.Q * reserve_ratio * (1.0 - t_ratio)
        env.hub_capacities = np.maximum(0.0, true_capacities - frozen_cap)

        next_obs, _, done, _, info = env.step(action)
        
        consumed = np.maximum(0.0, true_capacities - frozen_cap) - env.hub_capacities
        env.hub_capacities = true_capacities - consumed

        step_coverages.append(info['step_coverage'])
        obs = next_obs
            
    return np.array(step_coverages)

def run_rl_best(env, day_idx, predictor, ppo_policy, device):
    """运行 RL (Best) 策略"""
    env.current_scenario = env.eval_scenarios[day_idx]
    env.day_total_expected_demand = float(np.sum(env.current_scenario))
    env.current_t = 0
    env.hub_capacities = np.full(env.K, env.cfg.Q, dtype=np.float32)
    env.ep_total_demand = 0.0
    env.ep_total_unmet = 0.0
    
    total_initial_cap = env.K * env.cfg.Q
    env.day_static_target = min(1.0, total_initial_cap / max(env.day_total_expected_demand, 1.0))

    obs = env._get_obs()
    done = False
    
    initial_orders = obs['current_orders'].copy()
    history_buffer = deque([initial_orders for _ in range(12)], maxlen=12)
    step_coverages = []

    with torch.no_grad():
        while not done:
            history_buffer.append(obs['current_orders'])
            hist_tensor = torch.tensor(np.array(history_buffer), dtype=torch.float32, device=device) / ORDER_SCALE
            pred_orders_all = predictor(hist_tensor)
            predicted_orders = pred_orders_all[0, :] * ORDER_SCALE
            
            env.set_predictor_info(predicted_orders.sum().item())

            step_obs = {
                'node_features': torch.tensor(obs['node_features'], dtype=torch.float32, device=device),
                'current_orders': torch.tensor(obs['current_orders'], dtype=torch.float32, device=device),
                'hub_mask': torch.tensor(obs['hub_mask'], dtype=torch.float32, device=device),
                'hub_capacities': torch.tensor(obs['hub_capacities'], dtype=torch.float32, device=device),
                'predicted_orders': predicted_orders,
                'time_ratio': torch.tensor(obs['time_ratio'], dtype=torch.float32, device=device),
                'macro_pressure': torch.tensor(obs['macro_pressure'], dtype=torch.float32, device=device),
                'static_target': torch.tensor([env.day_static_target], dtype=torch.float32, device=device)
            }
            a_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)

            action_tensor, _, _, _ = ppo_policy.get_action(step_obs, action_mask=a_mask, deterministic=False)
            action = action_tensor.cpu().numpy()

            next_obs, _, done, _, info = env.step(action)
            step_coverages.append(info['step_coverage'])
            obs = next_obs
            
    return np.array(step_coverages)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UAVHubConfig()
    env = RobustHubEnv(cfg, mode='eval')
    
    # 🔥 发动攻击：把高峰提前 8 小时 (挪到大概早上 10点~下午2点)
    shift_scenarios_in_memory(env, shift_hours=-8)
    
    num_days = len(env.eval_scenarios)
    results_dict = {}

    # 1. 跑被戳中死穴的 Greedy (Reserve 30%)
    print(f"🏃 正在评估 Greedy (Reserve 30%) 面临突变时的表现...")
    greedy_covs = []
    for day in range(num_days):
        greedy_covs.append(run_greedy_reserve(env, day, 0.3))
    results_dict['Greedy (Reserve 30%)'] = np.array(greedy_covs)

    # 2. 跑自适应的 RL (Best)
    print(f"🤖 正在评估 RL (Best) 动态应对突变的能力...")
    model_dir = os.path.join(BASE_DIR, 'module_1_deployment', 'models')
    predictor = FutureDemandPredictor(N=env.N).to(device)
    ppo_policy = DynamicDispatchPPO(cfg, N=env.N, node_feature_dim=9, hidden_dim=128).to(device)
    
    predictor.load_state_dict(torch.load(os.path.join(model_dir, 'predictor_best.pth'), map_location=device, weights_only=True))
    ppo_policy.load_state_dict(torch.load(os.path.join(model_dir, 'ppo_policy_best.pth'), map_location=device, weights_only=True))
    predictor.eval()
    ppo_policy.eval()

    rl_covs = []
    for day in range(num_days):
        rl_covs.append(run_rl_best(env, day, predictor, ppo_policy, device))
    results_dict['RL (Best)'] = np.array(rl_covs)

    # 🎨 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    x_raw = np.linspace(0, 24, cfg.T_timesteps)
    x_smooth = np.linspace(0, 24, 300)

    for label, data_2d in results_dict.items():
        mean_curve = np.mean(data_2d, axis=0)
        spl_mean = make_interp_spline(x_raw, mean_curve, k=3)
        y_mean_smooth = spl_mean(x_smooth)
        
        color = '#9467bd' if 'Greedy' in label else '#ff7f0e'
        line_style = '--' if 'Greedy' in label else '-'
        
        ax.plot(x_smooth, y_mean_smooth, label=label, color=color, linestyle=line_style, linewidth=2.5)

    ax.set_title("灾难泛化测试：当需求高峰突变至中午 (Time-Shift Attack)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("一天的时间 (Hours, 0-24)", fontsize=13, labelpad=10)
    ax.set_ylabel("需求覆盖率 (Coverage Rate)", fontsize=13, labelpad=10)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(0, 1.05)
    
    # 画一个阴影区，标出中午突变的高峰期
    ax.axvspan(8, 14, color='gray', alpha=0.15, label='突发高峰期 (Shifted Peak)')

    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.legend(loc='lower left', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'time_shift_attack_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 65)
    print(f"✅ 攻击测试完成！绝杀图表已保存至:\n📁 {save_path}")

if __name__ == '__main__':
    main()