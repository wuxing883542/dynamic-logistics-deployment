"""
容量预留贪心基线 (Capacity Reservation Greedy)
用于证明 RL 的自适应动态分配能力，优于简单的启发式静态预留。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 确保能找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv

# 💡 建立清爽的独立输出文件夹
SAVE_DIR = os.path.join(BASE_DIR, 'data', 'greedy_reserve_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置中文字体 (防止图表中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def run_greedy_reserve(env, day_idx, reserve_ratio):
    """运行单天的预留贪心评估"""
    # ── 1. 强制重置与锁定当天场景 ──
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
    step_coverages = []

    while not done:
        # ── 2. 生成纯贪心动作 (永远寻找距离最近的合法枢纽) ──
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

        # ── 3. 💡 核心机制：运力隐身术 (动态冻结池) ──
        # 备份真实的物理运力
        true_capacities = env.hub_capacities.copy()
        
        # 计算当前时刻“必须被强制冻结”的保底运力 
        # (早上冻结最多，随着时间推移，晚高峰到来时冻结的运力逐渐解冻归零)
        t_ratio = env.current_t / env.T
        frozen_cap = env.cfg.Q * reserve_ratio * (1.0 - t_ratio)
        
        # 抛给环境的“可用运力” = 真实运力 - 冻结运力 (如果不够扣，就当做0)
        env.hub_capacities = np.maximum(0.0, true_capacities - frozen_cap)

        # ── 4. 与环境交互 ──
        next_obs, _, done, _, info = env.step(action)
        
        # ── 5. 结算真实运力 ──
        # 计算环境在“受限状态”下这一步真实用掉的运力
        consumed = np.maximum(0.0, true_capacities - frozen_cap) - env.hub_capacities
        # 从真实的物理运力池中扣除消耗，恢复系统真实状态
        env.hub_capacities = true_capacities - consumed

        # 记录本步业务指标
        step_coverages.append(info['step_coverage'])
        obs = next_obs
            
    return np.array(step_coverages)


def main():
    cfg = UAVHubConfig()
    # 强制开启 eval 模式，保证测试集的纯洁性
    env = RobustHubEnv(cfg, mode='eval')
    num_days = len(env.eval_scenarios)
    
    # 待测试的预留比例，0.0 就是无预留的纯贪心
    reserve_ratios = [0.0, 0.1, 0.2, 0.3]
    results_dict = {}

    print("=" * 65)
    print(f"🌍 开始运行容量预留贪心基线 (Capacity Reservation Greedy)")
    print(f"📦 评测数据集: EVAL | 总天数: {num_days}")
    print("=" * 65)

    for r in reserve_ratios:
        print(f"🏃 正在推演预留比例 {r*100:4.1f}% 的调度策略...")
        all_days_coverage = []
        for day in range(num_days):
            all_days_coverage.append(run_greedy_reserve(env, day, r))
        
        # 保存整个 eval 集的均值轨迹
        results_dict[f'Greedy (Reserve {r*100:.0f}%)'] = np.array(all_days_coverage)

    # ==========================================
    # 🎨 数据量化与可视化出图
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x_raw = np.linspace(0, 24, cfg.T_timesteps)
    x_smooth = np.linspace(0, 24, 300)

    # 经典学术配色
    colors = ['#FF4B4B', '#1f77b4', '#2ca02c', '#9467bd']
    
    print("\n" + "=" * 65)
    print("📊 终极量化结果报告 (论文级)：")
    print(f"{'调度策略':<25} | {'均覆盖 (Mean)':<12} | {'谷值托底 (Min)':<12} | {'时序方差 (Std)':<12}")
    print("-" * 65)

    for idx, (label, data_2d) in enumerate(results_dict.items()):
        # 计算整个测试集上的分时均值曲线
        mean_curve = np.mean(data_2d, axis=0)
        
        c_mean = np.mean(mean_curve)
        c_min = np.min(mean_curve)
        c_std = np.std(mean_curve)
        
        print(f"{label:<25} | {c_mean*100:>10.2f}% | {c_min*100:>10.2f}% | {c_std:>10.4f}")

        # 曲线平滑处理，增加论文质感
        spl_mean = make_interp_spline(x_raw, mean_curve, k=3)
        y_mean_smooth = spl_mean(x_smooth)
        
        # 无预留的纯贪心用虚线表示，其余用实线
        line_style = '--' if '0%' in label else '-'
        line_width = 2.5 if '0%' in label else 2.0
        
        ax.plot(x_smooth, y_mean_smooth, label=label, color=colors[idx], 
                linestyle=line_style, linewidth=line_width, alpha=0.9)

    # ── 图表细节打磨 ──
    ax.set_title("不同容量预留策略下的贪心分时覆盖率演进", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("一天的时间 (Hours, 0-24)", fontsize=13, labelpad=10)
    ax.set_ylabel("需求覆盖率 (Coverage Rate)", fontsize=13, labelpad=10)
    
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylim(0, 1.05)
    
    # 增加细致的背景网格
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'greedy_reserve_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 65)
    print(f"✅ 评估大功告成！对比图表已妥善保存至:\n📁 {save_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()