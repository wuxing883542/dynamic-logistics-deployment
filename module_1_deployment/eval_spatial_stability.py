import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── 1. 路径设置，确保能导入你的项目模块 ──
current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO


def analyze_spatial_stability(policy, env, device, num_trials=30, buffer_radius=300.0):
    """
    运行多次评估，分析选址的空间聚合度 (针对 2000x2000 曼哈顿网格的严格标准)
    """
    policy.eval()
    all_hubs_ids = []
    all_centroids = []
    
    print(f"\n🚀 开始进行 {num_trials} 次独立日场景的选址稳定性测算...")
    
    with torch.no_grad():
        for i in range(num_trials):
            obs, _ = env.reset()
            # 组装 Tensor (与 train_ppo.py 保持绝对对齐)
            t_obs = {
                'phase': obs['phase'],
                'node_features': torch.FloatTensor(obs['node_features']).unsqueeze(0).to(device),
                'hub_mask': torch.FloatTensor(obs['hub_mask']).unsqueeze(0).to(device),
                'hub_locations': torch.zeros((1, policy.K), dtype=torch.long).to(device)
            }
            
            # 使用确定性策略获取选址
            logits, _ = policy(t_obs, detach_backbone=True)
            hubs, _ = policy.sample_site(logits[0], deterministic=True)
            
            all_hubs_ids.append(hubs)
            
            # 获取这 K 个枢纽的真实物理坐标
            coords = env.coords[hubs]  # shape: (K, 2)
            centroid = coords.mean(axis=0) # 计算质心
            all_centroids.append(centroid)
            
    # ── 2. 计算质心漂移距离 ──
    all_centroids = np.array(all_centroids)
    mean_centroid = all_centroids.mean(axis=0)
    drifts = np.linalg.norm(all_centroids - mean_centroid, axis=1)
    avg_drift = drifts.mean()
    
    print("\n" + "="*50)
    print("📊 空间测算结果报告：")
    print(f"   平均质心漂移距离: {avg_drift:.2f} 米")
    
    # 采用严格的曼哈顿街区尺度 (单街区 280米)
    if avg_drift < 150:
        print("   ✅ 判定结论：选址在空间上 [高度聚合]！")
        print("      模型大盘极其稳定，几乎锁死了特定街区，找到了全局最优解！")
    elif avg_drift < 400:
        print("   ⚠️ 判定结论：选址在空间上 [存在微调]！")
        print("      模型正在聪明地跨街区（1~2个街区）适应不同日子的局部潮汐。")
    else:
        print("   ❌ 判定结论：选址 [尚未收敛]！")
        print("      大盘还在几个大区之间剧烈震荡，建议增加训练轮数或扩大 Batch Size。")
    print("="*50 + "\n")

    # ── 3. 画出炫酷的空间热力图 ──
    plt.figure(figsize=(10, 10))
    
    # 画出全城所有节点（浅灰色底图）
    plt.scatter(env.coords[:, 0], env.coords[:, 1], c='lightgray', s=40, label='All City Nodes')
    
    # 将历次选出来的枢纽画上去（半透明，重叠越多颜色越深）
    for hubs in all_hubs_ids:
        plt.scatter(env.coords[hubs, 0], env.coords[hubs, 1], c='red', s=400, alpha=0.15, edgecolors='none')
        
    plt.title(f"Spatial Hub Aggregation Heatmap\n(Avg Centroid Drift: {avg_drift:.1f}m)", fontsize=14, pad=15)
    plt.xlabel("X Coordinate (meters)", fontsize=12)
    plt.ylabel("Y Coordinate (meters)", fontsize=12)
    
    # 固定视野边界，匹配你的 2000x2000 地图
    plt.xlim(-100, 2100)
    plt.ylim(-100, 2100)
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    print("🎨 正在生成空间热力图，请查看弹出的窗口...")
    plt.show()


if __name__ == "__main__":
    # ── 4. 执行主干逻辑 ──
    cfg = UAVHubConfig()
    device = torch.device("cpu")  # 评估画图阶段用 CPU 即可，不占显存
    
    print("⚙️ 正在初始化环境和策略网络...")
    eval_env = RobustHubEnv(cfg, mode='eval')
    policy = GraphAttentionPPO(N=eval_env.N, node_dim=5, hidden_dim=128, K=cfg.max_hubs).to(device)
    
    # 读取你刚才训练出来保存在 models 里的最佳模型
    model_path = os.path.join(current_module_dir, "models", "best_eval_policy.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 严重错误：找不到模型权重文件！路径：{model_path}")
        print("请确认你已经成功运行了 train_ppo.py 并生成了 best_eval_policy.pth")
    else:
        # 加载权重
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型权重 [best_eval_policy.pth] 加载成功！")
        
        # 执行测算：针对 30 个不同的验证集日场景进行采样
        analyze_spatial_stability(policy, eval_env, device, num_trials=30)