import os
import sys
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO


def get_action_mask(hub_locations, max_hubs, N, device):
    """与训练时 get_action_mask 逻辑对齐：动作空间 max_hubs*(N+1)"""
    action_dim = max_hubs * (N + 1)
    mask = torch.ones(action_dim, dtype=torch.bool, device=device)
    occupied_nodes = set(h for h in hub_locations if h < N)

    for h_idx in range(max_hubs):
        own_node = hub_locations[h_idx] if h_idx < len(hub_locations) else N
        for node_idx in occupied_nodes:
            if node_idx == own_node:
                continue
            idx = h_idx * (N + 1) + node_idx
            mask[idx] = False
    return mask


def get_sinkhorn_routing(env, active_hubs, snap_demand):
    num_hubs = len(active_hubs)
    C_dist = np.zeros((env.N, num_hubs + 1))
    for j, h_idx in enumerate(active_hubs):
        distances = env.dist_matrix[:, h_idx]
        C_dist[:, j] = env.physics.calculate_economic_cost(distance=distances, demand=1.0)
    C_dist[:, -1] = 20.0  # 与训练时保持一致

    shadow_prices = np.zeros(num_hubs + 1)
    for _ in range(20):
        perceived_cost = C_dist + shadow_prices
        min_cost = np.min(perceived_cost, axis=1, keepdims=True)
        exp_cost = np.exp(-(perceived_cost - min_cost) / env.sinkhorn_temp)
        probs = exp_cost / np.sum(exp_cost, axis=1, keepdims=True)

        hub_loads = np.zeros(num_hubs)
        for j, h_idx in enumerate(active_hubs):
            mask = np.ones(env.N, dtype=bool)
            mask[h_idx] = False
            hub_loads[j] = np.sum(probs[mask, j] * snap_demand[mask])

        overloads = np.maximum(0, hub_loads - env.cfg.Q)
        shadow_prices[:-1] += overloads * env.shadow_step

    return probs


def evaluate_and_plot_model(model_name, model_filename, cfg, env, device):
    model_path = os.path.join(current_module_dir, 'models', model_filename)
    if not os.path.exists(model_path):
        print(f"⚠️ 跳过 {model_name}: 找不到模型文件 {model_path}")
        return

    print(f"\n=================================================")
    print(f"🧠 正在评估模型: {model_name}")
    print(f"=================================================")

    model = GraphAttentionPPO(N=cfg.N, max_hubs=cfg.max_hubs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ 权重 {name} 含有 NaN，模型已损坏。")
            return

    # 推理：逐步放置枢纽直到 episode 结束
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask = get_action_mask(env.hub_locations, cfg.max_hubs, cfg.N, device)

        with torch.no_grad():
            dist, _ = model(state_tensor)
            logits = dist.logits.squeeze(0)
            masked_logits = logits.masked_fill(~mask, -1e9)
            action = torch.argmax(masked_logits, dim=-1).item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    active_hubs = env.hub_locations  # list，不含虚拟节点
    print(f"✅ {model_name} 确定的最终枢纽节点: {active_hubs}")

    output_dir = os.path.join(project_root, 'data', f"{model_name}_Results")
    os.makedirs(output_dir, exist_ok=True)

    node_types = env.topo['node_types']
    coords = env.coords
    hub_colors = ['#FF4500', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2']

    period_info = [
        ("早高峰", 50),
        ("午高峰", 150),
        ("晚高峰", 250),
        ("深夜平峰", 350),
    ]

    # -------------------------------------------------------
    # 图1：2×2 拼图，四时段分配结果
    # -------------------------------------------------------
    period_assignments = []
    period_probs = []
    period_demands = []
    period_coverages = []

    for _, snap_idx in period_info:
        demand = env.snapshots[snap_idx]
        probs = get_sinkhorn_routing(env, active_hubs, demand)
        coverage = 1.0 - np.sum(probs[:, -1] * demand) / (np.sum(demand) + 1e-6)
        assignment = np.where(probs[:, -1] > 0.5, -1, np.argmax(probs[:, :-1], axis=1))
        period_assignments.append(assignment)
        period_probs.append(probs)
        period_demands.append(demand)
        period_coverages.append(coverage)

    drift_nodes = set()
    for i in range(env.N):
        valid_assignments = [a[i] for a in period_assignments if a[i] != -1]
        if len(valid_assignments) > 1 and len(set(valid_assignments)) > 1:
            drift_nodes.add(i)

    print(f"  🔀 归属漂移节点: {sorted(drift_nodes)}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f"[{model_name}] 潮汐需求下枢纽分配动态图\n"
                 f"枢纽节点: {active_hubs}  |  双圆圈节点 = 跨时期归属漂移",
                 fontsize=16, fontweight='bold', y=0.98)

    for ax_idx, (ax, (period_name, snap_idx)) in enumerate(zip(axes.flat, period_info)):
        demand = period_demands[ax_idx]
        probs = period_probs[ax_idx]
        assignment = period_assignments[ax_idx]
        coverage = period_coverages[ax_idx]

        for i in range(env.N):
            if assignment[i] == -1:
                continue
            hub_loc = active_hubs[assignment[i]]
            if i != hub_loc:
                line_weight = max(0.8, (probs[i, assignment[i]] * demand[i]) / 5.0)
                ax.plot([coords[i, 0], coords[hub_loc, 0]],
                        [coords[i, 1], coords[hub_loc, 1]],
                        color=hub_colors[assignment[i]], alpha=0.35,
                        linewidth=line_weight, zorder=1)

        for i in range(env.N):
            fill_color = '#D0EAF8' if node_types[i] == 0 else '#FDE8C8'
            label_text = f"N{i}\n{demand[i]:.0f}N"

            if assignment[i] == -1:
                ax.scatter(coords[i, 0], coords[i, 1], c='#AAAAAA', marker='X',
                           s=280, alpha=0.85, zorder=2)
                ax.text(coords[i, 0], coords[i, 1] + 12, label_text,
                        fontsize=7.5, ha='center', color='#888888')
            else:
                edge_color = hub_colors[assignment[i]]
                ax.scatter(coords[i, 0], coords[i, 1], c=fill_color, s=280,
                           edgecolors=edge_color, linewidths=2.5, zorder=2)
                if i in drift_nodes:
                    ax.scatter(coords[i, 0], coords[i, 1], c='none', s=520,
                               edgecolors='black', linewidths=1.5,
                               linestyles='--', zorder=2)
                ax.text(coords[i, 0], coords[i, 1] + 12, label_text,
                        fontsize=7.5, ha='center', fontweight='bold')

        for j, h_idx in enumerate(active_hubs):
            ax.scatter(coords[h_idx, 0], coords[h_idx, 1],
                       c=hub_colors[j], marker='*', s=800,
                       edgecolors='black', linewidths=1.5, zorder=4,
                       label=f'枢纽 {j} (节点{h_idx})')
            circle = plt.Circle((coords[h_idx, 0], coords[h_idx, 1]), 80,
                                 color=hub_colors[j], fill=False,
                                 linestyle='--', alpha=0.2, zorder=1)
            ax.add_artist(circle)

        ax.set_title(f"{period_name}  (快照#{snap_idx})  覆盖率: {coverage*100:.1f}%",
                     fontsize=13, pad=8)
        ax.set_xlim(-20, cfg.map_size + 20)
        ax.set_ylim(-20, cfg.map_size + 20)
        ax.set_xlabel("X 坐标 (m)", fontsize=10)
        ax.set_ylabel("Y 坐标 (m)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D0EAF8',
               markeredgecolor='gray', markersize=10, label='住宅区节点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FDE8C8',
               markeredgecolor='gray', markersize=10, label='商业区节点'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#AAAAAA',
               markersize=10, label='丢单节点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=12,
               markeredgewidth=1.5, linestyle='--', label='归属漂移节点'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_path = os.path.join(output_dir, "00_四时段分配对比图.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 四时段对比图已保存: {save_path}")

    # -------------------------------------------------------
    # 图2：归属稳定性热力图
    # -------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    stability = np.zeros(env.N)
    for i in range(env.N):
        valid_assignments = [a[i] for a in period_assignments if a[i] != -1]
        if len(valid_assignments) > 0:
            stability[i] = len(set(valid_assignments)) - 1

    avg_probs = np.mean(period_probs, axis=0)
    for i in range(env.N):
        best_hub_idx = np.argmax(avg_probs[i, :-1])
        hub_loc = active_hubs[best_hub_idx]
        if i != hub_loc and avg_probs[i, -1] <= 0.5:
            ax2.plot([coords[i, 0], coords[hub_loc, 0]],
                     [coords[i, 1], coords[hub_loc, 1]],
                     color='#CCCCCC', alpha=0.4, linewidth=1.0, zorder=1)

    cmap = plt.cm.RdYlGn_r
    for i in range(env.N):
        node_color = cmap(stability[i] / 3.0)
        size = 350 + stability[i] * 80
        ax2.scatter(coords[i, 0], coords[i, 1], c=[node_color], s=size,
                    edgecolors='black', linewidths=1.2, zorder=2, alpha=0.85)
        ax2.text(coords[i, 0], coords[i, 1] + 12, f"N{i}",
                 fontsize=8, ha='center',
                 fontweight='bold' if stability[i] > 0 else 'normal')

    for j, h_idx in enumerate(active_hubs):
        ax2.scatter(coords[h_idx, 0], coords[h_idx, 1],
                    c=hub_colors[j], marker='*', s=900,
                    edgecolors='black', linewidths=1.5, zorder=4,
                    label=f'枢纽 {j} (节点{h_idx})')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=3))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label('归属变化次数 (0=稳定, 3=高度漂移)', fontsize=11)
    cbar.set_ticks([0, 1, 2, 3])

    ax2.set_title(f"[{model_name}] 需求节点归属稳定性热力图\n颜色越红 = 跨时期归属变化越频繁",
                  fontsize=14, pad=12)
    ax2.set_xlim(-20, cfg.map_size + 20)
    ax2.set_ylim(-20, cfg.map_size + 20)
    ax2.set_xlabel("X 坐标 (m)", fontsize=11)
    ax2.set_ylabel("Y 坐标 (m)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.85)

    save_path2 = os.path.join(output_dir, "01_归属稳定性热力图.png")
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 归属稳定性热力图已保存: {save_path2}")
    print(f"🎉 {model_name} 全部图表已保存至: {output_dir}")


def visualize_and_save():
    cfg = UAVHubConfig()
    env = RobustHubEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_test = {
        "Best_Model": "best_gat_policy.pth",
        "Final_Model": "final_gat_policy.pth",
    }

    for model_name, model_filename in models_to_test.items():
        evaluate_and_plot_model(model_name, model_filename, cfg, env, device)


if __name__ == "__main__":
    visualize_and_save()
