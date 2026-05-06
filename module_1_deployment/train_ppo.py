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
# 作用：确保强化学习的勘探轨迹和网络初始化绝对一致。
# 答辩话术：保证课题实验结果具备严格的“可复现性 (Reproducibility)”。
# ==========================================
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 随机种子已锁定: {seed}")

current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO

# ==========================================
# 【物理硬约束：动作掩码 (Action Mask)】
# 作用：禁止系统在同一个地理节点上叠加建设多个枢纽。
# 凡是已被占用的节点，其动作概率被强制设为极大负数，彻底阻断非法搜索空间。
# ==========================================
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

def train():
    cfg = UAVHubConfig()
    set_global_seed(cfg.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练启动！当前设备: {device}")
    
    models_dir = os.path.join(current_module_dir, "models")
    logs_base_dir = os.path.join(current_module_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(logs_base_dir, f"run_{run_time}")
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_log_dir)

    env = RobustHubEnv(cfg)
    policy = GraphAttentionPPO(N=cfg.N, max_hubs=cfg.max_hubs).to(device)
    optimizer = Adam(policy.parameters(), lr=1e-4)

    # --------------------------------------------------
    # 【PPO 基础超参数】
    # --------------------------------------------------
    gamma = 0.99
    clip_epsilon = 0.2
    
    # 💡 恢复为 5：因为有了底层的 KL 动态早停作保镖，
    # 我们可以放心地让网络在安全的环境下跑满 5 次迭代，最大化前期爬坡速度！
    ppo_epochs = 5      
    max_episodes = 2000  
    best_reward = -float('inf')
    UPDATE_FREQ = 1  

    for episode in range(1, max_episodes + 1):
        
        # ==========================================================
        # 🏆 【核心亮点：探索与学习步长的解耦退火 (Decoupled Annealing)】
        # 痛点：传统 PPO 将学习率(lr)和探索率(entropy)同步衰减。在 1000 局后，
        # 探索率过低导致模型陷入局部最优，但此时较高的学习率反而会将模型推离好不容易找到的解，引发剧烈震荡。
        # 对策：提升探索率的底线，确保中后期依然具备逃逸局部最优的“好奇心”。
        # ==========================================================
        frac = 1.0 - (episode - 1.0) / max_episodes 
        
        # 学习率正常衰减，让步伐越迈越稳 (1e-4 -> 1e-5)
        current_lr = 1e-4 * frac + 1e-5 
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        # 探索率底线提升至 0.01 (0.05 -> 0.01)，解耦退火速度，防止“死脑筋”现象
        current_entropy_coef = 0.05 * frac + 0.01 
        
        # ==========================================================
        # 🛡️ 【核心亮点：动态自适应 KL 散度防线 (Dynamic KL Threshold)】
        # 逻辑：训练前期策略变动大，防线应宽容以鼓励试错 (0.015)；
        # 训练后期策略趋于平稳，防线应极其严苛以防止崩盘 (0.005)。
        # 完美解决固定阈值在后期失效、导致过度拟合的数学悖论。
        # ==========================================================
        current_target_kl = 0.015 * frac + 0.005

        state, _ = env.reset()
        done = False
        states_pool, actions_pool, log_probs_pool, values_pool, rewards_pool, masks_pool = [], [], [], [], [], []
        ep_reward = 0
        
        # --------------------------------------------------
        # 【阶段一：与极端潮汐环境交互，收集数据】
        # --------------------------------------------------
        while not done:
            states_pool.append(copy.deepcopy(state))
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, value = policy(state_tensor)
            logits = dist.logits.squeeze(0)
            
            mask = get_action_mask(state, cfg.max_hubs, cfg.N, device, env.hub_locations)
            masks_pool.append(mask)
            
            # 🛡️ 深度掩码保护：使用 -1e8 阻断无效动作，保护底层 float32 精度不出现 NaN
            masked_logits = logits.masked_fill(mask == False, -1e8)
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
        # 【阶段二：计算折扣回报与标准化优势函数 (Advantage)】
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
            
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        old_states = torch.FloatTensor(np.array(states_pool)).to(device)
        old_actions = torch.stack(actions_pool).to(device)
        old_log_probs = torch.stack(log_probs_pool).detach().to(device)
        old_values = torch.stack(values_pool).detach().to(device)
        old_masks = torch.stack(masks_pool).to(device)
        
        advantages = returns_tensor - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------
        # 【阶段三：PPO 网络核心更新与动态拦截】
        # --------------------------------------------------
        for _ in range(ppo_epochs):
            new_dist, new_values = policy(old_states)
            new_logits = new_dist.logits
            
            masked_new_logits = new_logits.masked_fill(old_masks == False, -1e8)
            new_safe_dist = Categorical(logits=masked_new_logits)
            new_log_probs = new_safe_dist.log_prob(old_actions)
            
            # 🏆 采用严格的平方 KL 散度近似：0.5 * (old - new)^2
            # 消除 (old - new) 带来的不对称数学陷阱，确保“防抱死系统”不会被负值骗过。
            with torch.no_grad():
                approx_kl = 0.5 * (old_log_probs - new_log_probs).pow(2).mean().item()
            
            # 触发动态早停 (Dynamic Early Stopping)：一旦策略偏移量超过当前严格的阈值，立刻中断保护大脑！
            if approx_kl > 1.5 * current_target_kl:
                break
                
            entropy = new_safe_dist.entropy().mean()
            
            # 底层截断防爆锁，防止 torch.exp() 产生无穷大
            log_ratio = torch.clamp(new_log_probs - old_log_probs, min=-20.0, max=5.0)
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(-1), returns_tensor)
            
            total_loss = actor_loss + 0.5 * critic_loss - current_entropy_coef * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        # --------------------------------------------------
        # 【阶段四：指标记录与终端账单级打印】
        # --------------------------------------------------
        writer.add_scalar("Train/Reward", ep_reward, episode)
        writer.add_scalar("Train/Cost_Total", info.get("avg_cost", 0), episode)
        writer.add_scalar("Metric/Coverage", info["avg_coverage"], episode)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(policy.state_dict(), os.path.join(models_dir, "best_gat_policy.pth"))
            print(f"⭐ Ep {episode:03d} | New Best Reward: {best_reward:.2f} | Total Cost: {info.get('avg_cost', 0):.0f} (Op: {info.get('operational_cost', 0):.0f}, Fix: {info.get('fixed_cost', 0):.0f}, Idle: {info.get('idle_penalty', 0):.0f}, Std: {info.get('std_op_cost', 0):.0f}) | Cov: {info['avg_coverage']*100:.1f}% | Hubs: {info['active_hubs']}")
            
        elif episode % 10 == 0:
            print(f"🎮 Ep {episode:03d} | Reward: {ep_reward:.2f} | Total Cost: {info.get('avg_cost', 0):.0f} (Op: {info.get('operational_cost', 0):.0f}, Fix: {info.get('fixed_cost', 0):.0f}, Idle: {info.get('idle_penalty', 0):.0f}, Std: {info.get('std_op_cost', 0):.0f}) | Cov: {info['avg_coverage']*100:.1f}% | Hubs: {info['active_hubs']}")

    final_model_path = os.path.join(models_dir, "final_gat_policy.pth")
    torch.save(policy.state_dict(), final_model_path)
    writer.close()
    
    print("✅ 训练完成！最强大脑已锁定。")

if __name__ == "__main__":
    train()