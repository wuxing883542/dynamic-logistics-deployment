import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from torch.optim.lr_scheduler import LinearLR
import random  # 💡 [新增] 导入 random
# 确保能找到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import DynamicDispatchPPO, FutureDemandPredictor
# ==========================================
# 💡 [新增] 全局随机种子，确保实验 100% 可复现
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# ==========================================
# 🚀 主训练循环 (最终极版：IPPO + CTDE + 宏微观影子价格)
# ==========================================
def train():
    set_seed(42)  # 💡 [新增] 第一时间锁定种子
    cfg = UAVHubConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")

    model_dir = os.path.join(MODULE_DIR, 'models')
    log_dir = os.path.join(MODULE_DIR, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = RobustHubEnv(cfg, mode='train')
    obs_sample, _ = env.reset()
    N = env.N
    
     # 💡 超参数设置
    total_episodes = 5000
    gamma = 0.995           # 必须使用长视距衰减，让 Critic 能看向未来
    gae_lambda = 0.95       
    clip_epsilon = 0.2      
    c_value = 0.5   

    # 💡 [修改] 设定探索熵的起点和终点
    c_entropy_start = 0.10  # 前期给高一点，鼓励多探索不同枢纽
    c_entropy_end = 0.03    # 后期降下来，让动作分布变尖锐，锁定最优解
    ppo_epochs = 4          

    ORDER_SCALE = 100.0


    # 💡 初始化双脑网络
    predictor = FutureDemandPredictor(N=N, history_len=12, pred_len=4, hidden_dim=64).to(device)
    # 节点特征维度为 9 (包含百分位优先级等)
    ppo_policy = DynamicDispatchPPO(cfg, N=N, node_feature_dim=9, hidden_dim=128).to(device)
    
    opt_pred = optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_ppo = optim.Adam(ppo_policy.parameters(), lr=3e-4, eps=1e-5)
    scheduler_ppo = LinearLR(opt_ppo, start_factor=1.0, end_factor=0.01, total_iters=total_episodes)

    scheduler_pred = LinearLR(opt_pred, start_factor=1.0, end_factor=0.01, total_iters=total_episodes)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'separated_inference_run'))

   

    print("=================================================")
    print("🚀 开始分离式双脑训练 (预测辅助 + 影子价格 + IPPO)")
    print("=================================================")

    recent_cov = deque(maxlen=20)
    recent_cost = deque(maxlen=20)

    best_eval_score = 0.0

    for episode in range(1, total_episodes + 1):
        # 💡 [新增] 线性衰减当前轮次的熵系数
        current_c_entropy = c_entropy_start - (c_entropy_start - c_entropy_end) * (episode / total_episodes)
        current_c_entropy = max(current_c_entropy, c_entropy_end) # 兜底机制
        
        obs, _ = env.reset()
        done = False

        # 冷启动边缘填充 (历史窗口初始化)
        initial_orders = obs['current_orders'].copy()
        history_buffer = deque([initial_orders for _ in range(12)], maxlen=12)

        rollout_data = {
            'obs_node': [], 'obs_orders': [], 'obs_mask': [], 'obs_cap': [], 'obs_pred': [],
            'obs_time': [], 'obs_macro': [], # 💡 [修复 Bug #4] 新增宏观压力收集
            'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'action_masks': [],
            'hist_windows': [], 'actual_next_orders': []
        }

        ep_penalty = 0.0
        ep_demand = 0.0
        ep_unmet = 0.0
        step_covs = []  # 记录每步局部覆盖率，用于衡量“时序公平性”

        predictor.eval()
        ppo_policy.eval()

        with torch.no_grad():
            while not done:
                # 1. 提取历史窗口并预测未来需求
                history_buffer.append(obs['current_orders'])
                hist_tensor = torch.tensor(np.array(history_buffer), dtype=torch.float32, device=device) / ORDER_SCALE
                pred_orders_all = predictor(hist_tensor) 
                predicted_orders = pred_orders_all[0, :] * ORDER_SCALE

                # 2. 提取未来 4 步全城预测总单量
                predicted_total_demand = predicted_orders.sum().item()

                # 3. 准备 PPO 观测数据
                o_node = torch.tensor(obs['node_features'], dtype=torch.float32, device=device)
                o_orders = torch.tensor(obs['current_orders'], dtype=torch.float32, device=device)
                o_mask = torch.tensor(obs['hub_mask'], dtype=torch.float32, device=device)
                o_cap = torch.tensor(obs['hub_capacities'], dtype=torch.float32, device=device)
                o_time = torch.tensor(obs['time_ratio'], dtype=torch.float32, device=device)
                o_macro = torch.tensor(obs['macro_pressure'], dtype=torch.float32, device=device) # 💡 获取宏观压力
                a_mask = torch.tensor(env.get_action_mask(), dtype=torch.bool, device=device)

                step_obs = {
                    'node_features': o_node,
                    'current_orders': o_orders,
                    'hub_mask': o_mask,
                    'hub_capacities': o_cap,
                    'predicted_orders': predicted_orders,
                    'time_ratio': o_time,
                    'macro_pressure': o_macro, # 💡 喂给 PPO 状态字典
                    'static_target': torch.tensor([env.day_static_target], dtype=torch.float32, device=device)
                }

                # 💡 [修复 Bug #3] 在调用 step 前，通过专用接口传入预测需求，保持 Gym 标准化
                env.set_predictor_info(predicted_total_demand)

                # 4. 策略网络输出动作
                action, log_prob, _, value = ppo_policy.get_action(step_obs, action_mask=a_mask, deterministic=False)

                # 💡 [修复 Bug #3] 恢复了最纯净的 step 签名
                next_obs, _, done, _, info = env.step(action.cpu().numpy())

                # 6. 存储 Rollout 序列
                rollout_data['obs_node'].append(o_node)
                rollout_data['obs_orders'].append(o_orders)
                rollout_data['obs_mask'].append(o_mask)
                rollout_data['obs_cap'].append(o_cap)
                rollout_data['obs_pred'].append(predicted_orders)
                rollout_data['obs_time'].append(o_time)
                rollout_data['obs_macro'].append(o_macro) # 💡 记录轨迹
                rollout_data['actions'].append(action)
                rollout_data['log_probs'].append(log_prob)
                rollout_data['values'].append(value)
                rollout_data['action_masks'].append(a_mask)
                
                # [IPPO] 接收节点独立核算的 Reward 数组
                node_rewards = torch.tensor(info['node_rewards'], dtype=torch.float32, device=device)
                rollout_data['rewards'].append(node_rewards) 
                
                rollout_data['hist_windows'].append(torch.tensor(np.array(history_buffer), dtype=torch.float32, device=device))
                rollout_data['actual_next_orders'].append(
                    torch.tensor(next_obs['current_orders'], dtype=torch.float32, device=device)
                )

                # 业务指标累加
                ep_penalty += info['unmet_penalty']
                ep_demand += info['step_demand']
                ep_unmet += info['step_unmet']
                step_covs.append(info['step_coverage'])

                obs = next_obs

        # ── 7. 多智能体 GAE 信用分配计算 ──
        rewards = rollout_data['rewards'] 
        values = rollout_data['values'] + [torch.zeros(N, device=device)] 
        
        advantages = torch.zeros((len(rewards), N), device=device) 
        
        last_gae_lam = torch.zeros(N, device=device)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
        
        returns = advantages + torch.stack(rollout_data['values'])
        
        # [防震荡终极修复] 废除减去均值的零和博弈陷阱，仅除以标准差控制方差！
        advantages = advantages / (advantages.std() + 1e-8)

        # ── 8. 批处理数据打包 ──
        b_obs = {
            'node_features': torch.stack(rollout_data['obs_node']),
            'current_orders': torch.stack(rollout_data['obs_orders']),
            'hub_mask': torch.stack(rollout_data['obs_mask']),
            'hub_capacities': torch.stack(rollout_data['obs_cap']),
            'predicted_orders': torch.stack(rollout_data['obs_pred']),
            'time_ratio': torch.stack(rollout_data['obs_time']),
            'macro_pressure': torch.stack(rollout_data['obs_macro']), # 💡 喂给联合更新计算
            'static_target': torch.full((len(rollout_data['obs_time']), 1), env.day_static_target, dtype=torch.float32, device=device),
        }
        b_actions = torch.stack(rollout_data['actions'])
        b_old_log_probs = torch.stack(rollout_data['log_probs']).detach() # (B, N)
        b_returns = returns.detach()                                      # (B, N)
        b_advantages = advantages.detach()                                # (B, N)
        b_action_masks = torch.stack(rollout_data['action_masks'])

        b_hist_windows = torch.stack(rollout_data['hist_windows'])
        b_actual_next = torch.stack(rollout_data['actual_next_orders'])

        # ── 9. 双脑网络联合更新 ──
        
        # Predictor 监督学习：单集只训 1 次严防过拟合
        predictor.train()
        opt_pred.zero_grad()
        pred_out = predictor(b_hist_windows / ORDER_SCALE) 
        pred_loss = nn.MSELoss()(pred_out[:, 0, :], b_actual_next / ORDER_SCALE) 
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.5) 
        opt_pred.step()

        # PPO 多智能体更新 (解绑 Advantage)
        ppo_policy.train()
        for _ in range(ppo_epochs):
            dist, new_values = ppo_policy(b_obs, action_mask=b_action_masks)

            new_log_probs = dist.log_prob(b_actions) # (B, N)
            ratio = torch.exp(new_log_probs - b_old_log_probs) # (B, N)

            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages
            
            # [IPPO核心] 节点级别的误差独立反馈，最后在整体求 mean
            actor_loss = -torch.min(surr1, surr2).mean() 

            critic_loss = c_value * nn.MSELoss()(new_values, b_returns)
            entropy_loss = dist.entropy().mean()
            
            ppo_loss = actor_loss + critic_loss - current_c_entropy * entropy_loss

            opt_ppo.zero_grad()
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo_policy.parameters(), max_norm=0.5)
            opt_ppo.step()

        # ── 10. 核心业务指标与 TensorBoard 写入 ──
        ep_cov = 1.0 - (ep_unmet / max(ep_demand, 1e-5))
        ep_cost = ep_penalty
        # ✅ 新增：获取跑完一整天后的系统总剩余运力
        ep_rem_capacity = float(np.sum(env.hub_capacities))
        # 记录单集整城所有节点获得的总奖赏
        ep_reward = sum([r.sum().item() for r in rollout_data['rewards']])

        recent_cov.append(ep_cov)
        recent_cost.append(ep_cost)

        writer.add_scalar("0_RL/Episode_Reward", ep_reward, episode)
        writer.add_scalar("1_Business/Coverage_Rate", ep_cov, episode)
        writer.add_scalar("1_Business/Total_Cost", ep_cost, episode)
        writer.add_scalar("1_Business/Penalty_Cost", ep_penalty, episode)
        writer.add_scalar("1_Business/Remaining_Capacity", ep_rem_capacity, episode)
        # 时序公平性的论文级核心指标：谷值覆盖率与全天方差
        writer.add_scalar("1_Business/StepCov_Min", min(step_covs), episode)
        writer.add_scalar("1_Business/StepCov_Std", np.std(step_covs), episode)

        writer.add_scalar("2_Loss/Predictor_MSE", pred_loss.item(), episode)
        writer.add_scalar("2_Loss/PPO_Actor", actor_loss.item(), episode)
        writer.add_scalar("2_Loss/PPO_Critic", critic_loss.item(), episode)
        writer.add_scalar("2_Loss/PPO_Entropy", entropy_loss.item(), episode)

        # 记录完 TensorBoard 后，让学习率衰减
        scheduler_ppo.step()
        scheduler_pred.step() # ✅ 新增：预测器学习率推进一步


        # ── 11. 控制台周期监控 ──
        if episode % 10 == 0:
            avg_cov = sum(recent_cov) / len(recent_cov)
            avg_cost = sum(recent_cost) / len(recent_cost)
            real_mse = pred_loss.item() * (ORDER_SCALE ** 2) 
            # 增加 StepCov_Min (谷值覆盖)，一眼盯紧晚高峰有没有崩盘！
            print(f"[Ep {episode:04d}] 奖赏: {ep_reward:7.1f} | 均覆盖: {avg_cov*100:5.2f}% | 谷值覆盖: {min(step_covs)*100:5.2f}% | 拒单: {ep_penalty:7.0f} | MSE: {real_mse:.1f}")

        # ==========================================
        # 💡 [新增] 巅峰权重保存逻辑 (Best Checkpoint)
        # ==========================================
        current_min_cov = min(step_covs)
        # 使用当前轮次的 ep_cov 作为评估标准，比使用平滑后的 avg_cov 更敏锐
        current_score = ep_cov + current_min_cov 
        
        # 必须同时突破两大门槛 (均值 > 80%, 谷值 > 45%)，且总分超越历史最佳
        if ep_cov > 0.80 and current_min_cov > 0.45 and current_score > best_eval_score:
            best_eval_score = current_score
            print(f"🌟 [巅峰突破] Ep {episode:04d} 创下新高！均覆盖: {ep_cov*100:.2f}%, 谷值: {current_min_cov*100:.2f}% (综合得分: {current_score:.4f})")
            
            # 独立保存为 best 权重，防止被 500 轮的常规保存覆盖
            torch.save(ppo_policy.state_dict(), os.path.join(model_dir, 'ppo_policy_best.pth'))
            torch.save(predictor.state_dict(), os.path.join(model_dir, 'predictor_best.pth'))


        if episode % 500 == 0:
            torch.save(ppo_policy.state_dict(), os.path.join(model_dir, f'ppo_policy_ep{episode}.pth'))
            torch.save(predictor.state_dict(), os.path.join(model_dir, f'predictor_ep{episode}.pth'))

    print("✅ 训练完成！")
    writer.close()

if __name__ == '__main__':
    train()