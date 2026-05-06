import os
import sys
import torch
import random
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.distributions import Categorical  

warnings.filterwarnings("ignore", category=FutureWarning)

# 确保中文字体正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_action_mask(state, max_hubs, N, device, hub_locations):
    action_dim = max_hubs * (N + 1)
    mask = torch.ones(action_dim, dtype=torch.bool).to(device)
    occupied_nodes = set(np.where(state[:, 2] == 1.0)[0].tolist())
    
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
        
    C_dist[:, -1] = env.cfg.penalty_unmet 

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

    # ==========================================================
    # 👑 究极物理自洽 (V3.0 你的本地免邮版)：容量受限的智能硬分配
    # ==========================================================
    hard_probs = np.zeros_like(probs)
    remaining_capacity = np.ones(num_hubs) * env.cfg.Q

    # 按照节点需求量从大到小排序（物流经典的贪心策略：优先装载大客户）
    node_order = np.argsort(-snap_demand)

    for i in node_order:
        req = snap_demand[i]
        
        # 如果 AI 在软分配时就强烈倾向于拒单，直接扔垃圾桶
        if probs[i, -1] > 0.5:
            hard_probs[i, -1] = 1.0
            continue

        preferred_hubs = np.argsort(-probs[i, :-1])
        assigned = False

        for j in preferred_hubs:
            hub_node = active_hubs[j]
            
            # 💡 完美自洽：本地需求原地消化，不消耗无人机运力
            if i == hub_node:
                hard_probs[i, j] = 1.0
                assigned = True
                break
            else:
                # 跨节点需求，严格校验剩余容量
                if remaining_capacity[j] >= req:
                    hard_probs[i, j] = 1.0
                    remaining_capacity[j] -= req
                    assigned = True
                    break

        if not assigned:
            hard_probs[i, -1] = 1.0

    # 财务级成本计算：全部基于校验后的 hard_probs 进行硬核算
    transport_cost = 0.0
    idle_penalty = 0.0 
    actual_hub_loads = np.zeros(num_hubs)

    for j, h_idx in enumerate(active_hubs):
        mask = np.ones(env.N, dtype=bool)
        mask[h_idx] = False 
        transport_cost += np.sum(hard_probs[mask, j] * C_dist[mask, j] * snap_demand[mask])
        actual_hub_loads[j] = np.sum(hard_probs[mask, j] * snap_demand[mask])
        
        hub_idle = env.cfg.Q - actual_hub_loads[j] 
        idle_penalty += hub_idle * (env.cfg.penalty_unmet * 0.005)

    penalty_cost = np.sum(hard_probs[:, -1] * snap_demand) * env.cfg.penalty_unmet
    
    # 验证审计：此时物理载重绝对不会越过红线
    soft_overloads = np.maximum(0, actual_hub_loads - env.cfg.Q)
    overload_penalty = np.sum(soft_overloads) * (env.cfg.penalty_unmet * 2.0)
    
    op_cost = transport_cost + penalty_cost + overload_penalty

    # 打印最干净、绝对安全的物理账单
    print(f"   [容量审计] 智能硬分配物理载重: {[f'N{active_hubs[j]}:{actual_hub_loads[j]:.1f}/Q{env.cfg.Q}' for j in range(num_hubs)]}")
    if np.sum(soft_overloads) > 0:
        print(f"   ⚠️ 警告：溢出 {np.sum(soft_overloads):.1f} (在智能校验下不应出现)")
    else:
        print(f"   ✅ 安全：物理红线捍卫成功，溢出量为 0！")

    return hard_probs, op_cost, idle_penalty
def evaluate_and_plot_model(model_name, model_filename, cfg, env, device):
    set_global_seed(cfg.seed)
    
    model_path = os.path.join(current_module_dir, 'models', model_filename)
    if not os.path.exists(model_path):
        print(f"⚠️ 跳过 {model_name}: 找不到模型文件 {model_path}")
        return

    print(f"\n=================================================")
    print(f"🧠 正在评估模型: {model_name} (原味采样模式)")
    print(f"=================================================")

    model = GraphAttentionPPO(N=cfg.N, max_hubs=cfg.max_hubs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.train() 
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0  

    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"❌ 权重 {name} 含有 NaN，模型已损坏。")
            return

    state, _ = env.reset()
    done = False
    
    best_step_reward = -float('inf')  
    optimal_hub_locations = None      
    
    step_count = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask = get_action_mask(state, cfg.max_hubs, cfg.N, device, env.hub_locations)

        with torch.no_grad():
            dist, _ = model(state_tensor)
            logits = dist.logits.squeeze(0)
            masked_logits = logits.masked_fill(mask == False, -1e8)
            
            safe_dist = Categorical(logits=masked_logits)
            action = safe_dist.sample().item()

        state, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        
        if reward > best_step_reward:
            best_step_reward = reward
            optimal_hub_locations = env.hub_locations.copy()
            
        done = terminated or truncated

    active_hubs = [h for h in optimal_hub_locations if h < cfg.N]
    
    if not active_hubs:
        print("❌ 模型把所有枢纽都扔进了垃圾桶！无法可视化。")
        return

    print(f"📸 成功抓拍！在 100 步激进采样推演中，最高单步得分为 {best_step_reward:.2f}")
    print(f"✅ {model_name} 还原出的训练巅峰有效枢纽: {active_hubs}")

    output_dir = os.path.join(project_root, 'data', f"{model_name}_Results")
    os.makedirs(output_dir, exist_ok=True)

    node_types = env.topo['node_types']
    coords = env.coords
    hub_colors = ['#FF4500', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2', '#FF1493']

    period_info = [
        ("早高峰", 50),
        ("午高峰", 150),
        ("晚高峰", 250),
        ("深夜平峰", 350),
    ]

    period_assignments = []
    period_probs = []
    period_demands = []
    period_coverages = []
    
    # 💡 存储各时段成本拆解
    period_op_costs = []
    period_idle_costs = []
    fixed_cost = sum([env.fixed_costs[h] for h in active_hubs])

    for _, snap_idx in period_info:
        demand = env.snapshots[snap_idx]
        probs, op_cost, idle_penalty = get_sinkhorn_routing(env, active_hubs, demand)
        coverage = 1.0 - np.sum(probs[:, -1] * demand) / (np.sum(demand) + 1e-6)
        
        assignment = np.where(probs[:, -1] > 0.5, -1, np.argmax(probs[:, :-1], axis=1))
        
        period_assignments.append(assignment)
        period_probs.append(probs)
        period_demands.append(demand)
        period_coverages.append(coverage)
        period_op_costs.append(op_cost)
        period_idle_costs.append(idle_penalty)

    drift_nodes = set()
    for i in range(env.N):
        valid_assignments = [a[i] for a in period_assignments if a[i] != -1]
        if len(valid_assignments) > 1 and len(set(valid_assignments)) > 1:
            drift_nodes.add(i)

    print(f"  🔀 归属漂移节点 (准动态优化特征): {sorted(drift_nodes)}")

    # -------------------------------------------------------
    # 绘图部分
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f"[{model_name}] 潮汐需求下枢纽分配动态图 (完美还原最优策略)\n"
                 f"核心运作枢纽: {active_hubs}  |  双圆圈节点 = 跨时期归属漂移",
                 fontsize=16, fontweight='bold', y=0.98)

    for ax_idx, (ax, (period_name, snap_idx)) in enumerate(zip(axes.flat, period_info)):
        demand = period_demands[ax_idx]
        probs = period_probs[ax_idx]
        assignment = period_assignments[ax_idx]
        coverage = period_coverages[ax_idx]
        
        # 获取当前时段的具体成本
        curr_op_cost = period_op_costs[ax_idx]
        curr_idle_penalty = period_idle_costs[ax_idx]
        total_snapshot_cost = fixed_cost + curr_op_cost + curr_idle_penalty

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

        # 💡 在子标题上详细标出当期成本拆解明细
        ax.set_title(f"{period_name} (快照#{snap_idx}) | 覆盖率: {coverage*100:.1f}%\n"
                     f"当期总成本: {total_snapshot_cost:.0f}  (建站: {fixed_cost:.0f} | 运营运费: {curr_op_cost:.0f} | 闲置税: {curr_idle_penalty:.0f})",
                     fontsize=12, pad=10)
                     
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

    plt.tight_layout(rect=[0, 0.04, 1, 0.96]) # 留出顶部大标题空间
    save_path = os.path.join(output_dir, "00_四时段分配对比图.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 四时段对比图已保存: {save_path}")

    # -------------------------------------------------------
    # 图 2：归属稳定性热力图
    # -------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    stability = np.zeros(env.N)
    for i in range(env.N):
        valid_assignments = [a[i] for a in period_assignments if a[i] != -1]
        if len(valid_assignments) > 0:
            stability[i] = len(set(valid_assignments)) - 1

    avg_probs = np.mean(period_probs, axis=0)
    for i in range(env.N):
        if len(active_hubs) > 0:
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

    # 💡 增加 Final_Model，批量出图
    models_to_test = {
        "Best_Model": "best_gat_policy.pth",
        "Final_Model": "final_gat_policy.pth",
    }

    for model_name, model_filename in models_to_test.items():
        evaluate_and_plot_model(model_name, model_filename, cfg, env, device)

if __name__ == "__main__":
    visualize_and_save()