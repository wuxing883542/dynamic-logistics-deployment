"""
分时覆盖率曲线对比: RL vs Greedy (论文终极版)
包含：随机种子固化、小时制X轴、平滑曲线、误差置信带、量化指标柱状图
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from collections import deque
from scipy.interpolate import make_interp_spline

# 确保能找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import DynamicDispatchPPO, FutureDemandPredictor

# ==========================================
# ⚙️ 核心评测配置区
# ==========================================
DATASET_MODE = 'train' #eval或train
TARGET_EPISODES = [1000, 1500, 2000, 2500] 
ORDER_SCALE = 100.0

# 💡 1. 增加全局随机种子，确保实验绝对可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 设置中文字体 (若无字体可注释)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def run_simulation(env, day_idx, models=None, device='cpu'):
    # 强制固定场景并重置环境
    scenarios = env.train_scenarios if DATASET_MODE == 'train' else env.eval_scenarios
    env.current_scenario = scenarios[day_idx]
    env.day_total_expected_demand = float(np.sum(env.current_scenario))
    env.current_t = 0
    env.hub_capacities = np.full(env.K, env.cfg.Q, dtype=np.float32)
    env.ep_total_demand = 0.0
    env.ep_total_unmet = 0.0
    env.current_predicted_demand = 0.0

    obs = env._get_obs()
    done = False
    
    initial_orders = obs['current_orders'].copy()
    history_buffer = deque([initial_orders for _ in range(12)], maxlen=12)
    
    step_coverages = []

    if models is not None:
        predictor, ppo_policy = models
        predictor.eval()
        ppo_policy.eval()

    with torch.no_grad():
        while not done:
            if models is None:
                # ── 贪心基线 (Greedy) ──
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
            else:
                # ── RL 策略 ──
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
                    'macro_pressure': torch.tensor(obs['macro_pressure'], dtype=torch.float32, device=device)
                }
                a_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)
                
                action_tensor, _, _, _ = ppo_policy.get_action(step_obs, action_mask=a_mask, deterministic=False)
                action = action_tensor.cpu().numpy()

            next_obs, _, done, _, info = env.step(action)
            step_coverages.append(info['step_coverage'])
            obs = next_obs
            
    return np.array(step_coverages)


def main():
    set_seed(42)  # 固定种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UAVHubConfig()
    env = RobustHubEnv(cfg, mode=DATASET_MODE)
    
    num_days = len(env.eval_scenarios) if DATASET_MODE == 'eval' else len(env.train_scenarios)
    print(f"🌍 开始进行时序公平性评估 | 数据集: {DATASET_MODE.upper()} | 总天数: {num_days}")

    # 💡 修改：这里不再存均值，而是存所有的二维数组 (Days, 96)，用于计算标准差
    results_dict = {}

    print("🏃 正在运行 Greedy 基准测试...")
    greedy_all_days = []
    for day in range(num_days):
        greedy_all_days.append(run_simulation(env, day, models=None, device=device))
    results_dict['Greedy'] = np.array(greedy_all_days)

    model_dir = os.path.join(BASE_DIR, 'module_1_deployment', 'models')
    
    for ep in TARGET_EPISODES:
        print(f"🤖 正在评估 RL 策略 (Ep {ep})...")
        predictor_path = os.path.join(model_dir, f'predictor_ep{ep}.pth')
        ppo_path = os.path.join(model_dir, f'ppo_policy_ep{ep}.pth')
        
        if not os.path.exists(predictor_path) or not os.path.exists(ppo_path):
            print(f"   ⚠️ 警告: 找不到 Ep {ep} 的权重，跳过。")
            continue
            
        predictor = FutureDemandPredictor(N=env.N).to(device)
        ppo_policy = DynamicDispatchPPO(cfg, N=env.N, node_feature_dim=9, hidden_dim=128).to(device)
        
        predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))
        ppo_policy.load_state_dict(torch.load(ppo_path, map_location=device, weights_only=True))
        
        rl_all_days = []
        for day in range(num_days):
            rl_all_days.append(run_simulation(env, day, models=(predictor, ppo_policy), device=device))
            
        results_dict[f'RL (Ep {ep})'] = np.array(rl_all_days)

    # ==========================================
    # 🎨 绘图：双子图布局 (左边曲线，右边量化柱状图)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1.2]})
    
    # 💡 2. X轴转为小时 (0-24)
    x_raw = np.linspace(0, 24, cfg.T_timesteps)
    x_smooth = np.linspace(0, 24, 300) # 用于平滑的更密集的点

    colors = ['#FF4B4B', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    
    labels = []
    means = []
    mins = []
    stds = []

    for idx, (label, data_2d) in enumerate(results_dict.items()):
        mean_curve = np.mean(data_2d, axis=0)
        std_curve = np.std(data_2d, axis=0)
        
        # 记录柱状图数据 (以均值曲线为基准计算宏观指标)
        labels.append(label)
        means.append(np.mean(mean_curve))
        mins.append(np.min(mean_curve))
        stds.append(np.std(mean_curve)) # 这里反映的是一天之内的波动率(越小越好)

        # 💡 3. 平滑处理
        spl_mean = make_interp_spline(x_raw, mean_curve, k=3)
        y_mean_smooth = spl_mean(x_smooth)
        
        

        # 绘制主曲线与误差带
        if label == 'Greedy':
            ax1.plot(x_smooth, y_mean_smooth, label=label, color='red', linestyle='--', linewidth=2.5, zorder=10)
            
        else:
            ax1.plot(x_smooth, y_mean_smooth, label=label, color=colors[idx], linewidth=2.0)
            

    # ── 左图 (ax1) 格式设置 ──
    ax1.set_title(f"日内分时覆盖率与鲁棒性对比 (数据集: {DATASET_MODE.upper()})", fontsize=15)
    ax1.set_xlabel("一天的时间 (Hours, 0-24)", fontsize=13)
    ax1.set_ylabel("需求覆盖率 (Coverage Rate)", fontsize=13)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2)) # 每 2 小时一个刻度
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=11)

    # ── 右图 (ax2) 量化柱状图绘制 ──
    x_pos = np.arange(len(labels))
    width = 0.25

    # 画三组柱子
    ax2.bar(x_pos - width, means, width, label='Mean (均覆盖率↑)', color='#4c72b0', alpha=0.9)
    ax2.bar(x_pos, mins, width, label='Min (谷值托底↑)', color='#dd8452', alpha=0.9)
    ax2.bar(x_pos + width, stds, width, label='Std (时序方差↓)', color='#55a868', alpha=0.9)

    ax2.set_title("时序公平性量化指标", fontsize=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, 'data', f'temporal_fairness_paper_{DATASET_MODE}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 评估完成！论文级图表已保存至: {save_path}")

if __name__ == '__main__':
    main()