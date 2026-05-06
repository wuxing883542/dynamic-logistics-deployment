import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import datetime

# ==========================================
# 【核心防御 1：全局随机种子锁定】
# ==========================================
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 GPU 卷积等操作是确定性的，保证实验绝对可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 随机种子已锁定: {seed}")

# ==========================================
# 【核心防御 2：跨目录寻址】
# ==========================================
current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置、环境和图注意力大脑
from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO

def get_action_mask(state, max_hubs, N, device, hub_locations):
    """
    防撞车面具：防止无人机枢纽被部署到已被占用的节点上。
    """
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

def train():
    cfg = UAVHubConfig()
    set_global_seed(cfg.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练启动！当前设备: {device}")
    
    # ==========================================
    # 【日志与模型存储路径配置】
    # ==========================================
    models_dir = os.path.join(current_module_dir, "models")
    logs_base_dir = os.path.join(current_module_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    
    # 💡 [极其关键]：按时间戳生成独立的日志文件夹，彻底杜绝 TensorBoard 曲线打结！
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(logs_base_dir, f"run_{run_time}")
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_log_dir)

    # 初始化环境与网络
    env = RobustHubEnv(cfg)
    policy = GraphAttentionPPO(N=cfg.N, max_hubs=cfg.max_hubs).to(device)
    optimizer = Adam(policy.parameters(), lr=1e-4)

    # PPO 超参数
    gamma = 0.99
    clip_epsilon = 0.2
    ppo_epochs = 5      
    max_episodes = 2000  
    entropy_coef = 0.05 
    best_reward = -float('inf')

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        done = False
        states_pool, actions_pool, log_probs_pool, values_pool, rewards_pool, masks_pool = [], [], [], [], [], []
        ep_reward = 0
        
        # --------------------------------------------------
        # 【阶段一：环境交互采样 (Rollout)】
        # --------------------------------------------------
        while not done:
            states_pool.append(copy.deepcopy(state))
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, value = policy(state_tensor)
            logits = dist.logits.squeeze(0)
            
            mask = get_action_mask(state, cfg.max_hubs, cfg.N, device, env.hub_locations)
            masks_pool.append(mask)
            
            # 🛡️ 医用级防爆：使用 -1e7 代替 -1e9，绝对保护底层 float32 梯度，杜绝 NaN
            masked_logits = logits.masked_fill(mask == False, -1e7)
            safe_dist = Categorical(logits=masked_logits)
            
            action = safe_dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated 
            
            actions_pool.append(action)
            log_probs_pool.append(safe_dist.log_prob(action))
            values_pool.append(value.squeeze())
            rewards_pool.append(reward)
            state = next_state
            ep_reward += reward
            
        # --------------------------------------------------
        # 【阶段二：计算优势函数 (Advantage Estimation)】
        # --------------------------------------------------
        returns = []
        if truncated:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, next_value = policy(next_state_tensor)
                discounted_r = next_value.item()
        else:
            discounted_r = 0.0 

        for r in reversed(rewards_pool):
            discounted_r = r + gamma * discounted_r
            returns.insert(0, discounted_r)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        old_states = torch.FloatTensor(np.array(states_pool)).to(device)
        old_actions = torch.stack(actions_pool).to(device)
        old_log_probs = torch.stack(log_probs_pool).detach().to(device)
        old_values = torch.stack(values_pool).detach().to(device)
        old_masks = torch.stack(masks_pool).to(device)
        
        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------
        # 【阶段三：PPO 核心网络更新 (Policy Update)】
        # --------------------------------------------------
        for _ in range(ppo_epochs):
            new_dist, new_values = policy(old_states)
            new_logits = new_dist.logits
            
            # 🛡️ 医用级防爆同步更新
            masked_new_logits = new_logits.masked_fill(old_masks == False, -1e7)
            new_safe_dist = Categorical(logits=masked_new_logits)
            
            new_log_probs = new_safe_dist.log_prob(old_actions)
            entropy = new_safe_dist.entropy().mean()
            
            # 🛡️ PPO 终极防爆锁：限制概率比差值，彻底防止 torch.exp() 产生 inf 或 NaN
            log_ratio = torch.clamp(new_log_probs - old_log_probs, min=-20.0, max=5.0)
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            
            total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        # --------------------------------------------------
        # 【阶段四：写入 TensorBoard】
        # --------------------------------------------------
        writer.add_scalar("Train/Reward", ep_reward, episode)
        writer.add_scalar("Train/Cost_Total", info.get("avg_cost", 0), episode)
        writer.add_scalar("Metric/Coverage", info["avg_coverage"], episode)

        # --------------------------------------------------
        # 【阶段五：日志打印与最优模型存档】
        # --------------------------------------------------
        # 🔥 更新终端输出格式：清晰展示总成本 = 建站 + 运费 + 闲置税 + 波动惩罚
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(policy.state_dict(), os.path.join(models_dir, "best_gat_policy.pth"))
            print(f"⭐ Ep {episode:03d} | New Best Reward: {best_reward:.2f} | Total Cost: {info.get('avg_cost', 0):.0f} (Op: {info.get('operational_cost', 0):.0f}, Fix: {info.get('fixed_cost', 0):.0f}, Idle: {info.get('idle_penalty', 0):.0f}, Std: {info.get('std_op_cost', 0):.0f}) | Cov: {info['avg_coverage']*100:.1f}% | Hubs: {info['active_hubs']}")
            
        elif episode % 10 == 0:
            print(f"🎮 Ep {episode:03d} | Reward: {ep_reward:.2f} | Total Cost: {info.get('avg_cost', 0):.0f} (Op: {info.get('operational_cost', 0):.0f}, Fix: {info.get('fixed_cost', 0):.0f}, Idle: {info.get('idle_penalty', 0):.0f}, Std: {info.get('std_op_cost', 0):.0f}) | Cov: {info['avg_coverage']*100:.1f}% | Hubs: {info['active_hubs']}")

    final_model_path = os.path.join(models_dir, "final_gat_policy.pth")
    torch.save(policy.state_dict(), final_model_path)
    writer.close()
    
    print("✅ 训练完成！最强大脑已锁定，可以执行 visualize_hub.py 进行出图验证。")

if __name__ == "__main__":
    train()