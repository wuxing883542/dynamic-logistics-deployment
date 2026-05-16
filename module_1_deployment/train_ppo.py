import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import datetime


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed locked: {seed}")


current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv
from module_1_deployment.model_gat_ppo import GraphAttentionPPO


def clone_obs(obs):
    return {k: v.clone().detach() if isinstance(v, torch.Tensor) else v
            for k, v in obs.items() if k != 'lstm_hidden'}


def obs_to_tensors(obs, device):
    return {
        'phase':           obs['phase'],
        'node_features':   torch.FloatTensor(obs['node_features']).unsqueeze(0).to(device),
        'current_orders':  torch.FloatTensor(obs['current_orders']).unsqueeze(0).to(device),
        'hub_mask':        torch.FloatTensor(obs['hub_mask']).unsqueeze(0).to(device),
        'hub_capacities':  torch.FloatTensor(obs['hub_capacities']).unsqueeze(0).to(device),
        'lstm_hidden':     None,
    }


def deterministic_eval(env, policy, device):
    obs, _ = env.reset()
    t_obs = obs_to_tensors(obs, device)
    logits, _ = policy(t_obs)
    hubs, _ = policy.sample_site(logits[0], deterministic=True)
    obs, reward_site, _, _, info_site = env.step(np.array(hubs))

    total_cost = -reward_site * 100.0
    total_transport = 0.0
    total_penalty = 0.0

    done = False
    lstm_hidden = None
    while not done:
        t_obs = obs_to_tensors(obs, device)
        t_obs['lstm_hidden'] = lstm_hidden
        logits, _, lstm_hidden = policy(t_obs)
        actions, _, _ = policy.sample_dispatch(logits, deterministic=True)
        obs, reward, done, _, info = env.step(actions.cpu().numpy())
        total_transport += info['transport_cost']
        total_penalty += info['unmet_penalty']

    total_cost += total_transport + total_penalty
    coverage = 1.0 - total_penalty / max(total_transport + total_penalty, 1.0)

    return {
        'avg_cost':     total_cost,
        'avg_coverage': coverage,
        'hub_locations': hubs,
        'fixed_cost':   info_site.get('fixed_cost', 0),
        'transport':    total_transport,
        'penalty':      total_penalty,
    }


# ── PPO update ─────────────────────────────────────────
def ppo_update(trajectories, policy, optimizer, device,
               gamma, gae_lambda, clip_epsilon, value_clip_epsilon,
               ppo_epochs, target_kl, entropy_coef, writer, episode):

    # Compute site-step GAE: the site step's return = discounted sum of all downstream rewards
    site_advs = []
    site_rets = []
    for traj in trajectories:
        ret = 0.0
        for s in reversed(traj):
            ret = s['reward'] + gamma * ret
        site_val = traj[0]['value'].item()  # Critic value at site step
        site_advs.append(ret - site_val)
        site_rets.append(ret)

    # Pre-normalize site advantages
    site_adv_t = torch.tensor(site_advs, dtype=torch.float32).to(device)
    if len(site_adv_t) > 1:
        site_adv_t = (site_adv_t - site_adv_t.mean()) / (site_adv_t.std() + 1e-8)
    site_ret_t = torch.tensor(site_rets, dtype=torch.float32).to(device)

    for _epoch in range(ppo_epochs):
        total_site_actor = 0.0
        total_disp_actor = 0.0
        total_site_critic = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_site = 0
        n_disp = 0

        for ti, traj in enumerate(trajectories):
            # ── Site step (GAE + Critic) ──
            s = traj[0]
            logits, new_value = policy(s['obs'])
            new_lp = policy.compute_site_log_prob(logits[0], s['action'])

            old_lp = s['log_prob'].detach()
            adv_i = site_adv_t[ti]
            ret_i = site_ret_t[ti]

            log_ratio = new_lp - old_lp
            ratio = torch.exp(log_ratio)
            kl = ((ratio - 1) - log_ratio).item()

            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_i
            site_actor = -torch.min(surr1, surr2)

            val_clipped = s['value'] + torch.clamp(new_value - s['value'], -value_clip_epsilon, value_clip_epsilon)
            v_loss = 0.5 * torch.max((new_value - ret_i)**2, (val_clipped - ret_i)**2)

            total_site_actor += site_actor
            total_site_critic += v_loss
            total_kl += kl
            n_site += 1

            # ── Dispatch steps (per-node cost advantage, no Critic) ──
            lstm_hidden = None
            for s in traj[1:]:
                obs = s['obs']
                obs['lstm_hidden'] = lstm_hidden
                logits, _, lstm_hidden = policy(obs)

                new_per_lp, _ = policy.compute_dispatch_log_prob(logits, s['action'].unsqueeze(0))
                # new_per_lp: (N,), old_per_lp: (N,), per_node_adv: (N,)

                old_per_lp = s['per_node_lp'].detach()
                per_node_adv = s['per_node_adv']

                log_ratio_i = new_per_lp - old_per_lp
                ratio_i = torch.exp(log_ratio_i)
                kl_i = ((ratio_i - 1) - log_ratio_i).mean().item()

                surr1_i = ratio_i * per_node_adv
                surr2_i = torch.clamp(ratio_i, 1 - clip_epsilon, 1 + clip_epsilon) * per_node_adv
                disp_actor = -torch.min(surr1_i, surr2_i).mean()

                from torch.distributions import Categorical
                entropy = Categorical(logits=logits).entropy().mean()

                total_disp_actor += disp_actor
                total_entropy += entropy
                total_kl += kl_i
                n_disp += 1

        n_total = max(n_site + n_disp, 1)
        avg_kl = total_kl / n_total

        if avg_kl > 1.5 * target_kl:
            break

        loss = (total_site_actor / max(n_site, 1) +
                total_disp_actor / max(n_disp, 1) +
                total_site_critic / max(n_site, 1))
        loss = loss - entropy_coef * (total_entropy / max(n_disp, 1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

    writer.add_scalar("PPO/site_actor_loss", (total_site_actor / max(n_site, 1)).item(), episode)
    writer.add_scalar("PPO/disp_actor_loss", (total_disp_actor / max(n_disp, 1)).item(), episode)
    writer.add_scalar("PPO/site_critic_loss", (total_site_critic / max(n_site, 1)).item(), episode)
    writer.add_scalar("PPO/approx_kl", avg_kl, episode)
    writer.add_scalar("PPO/entropy_dispatch", total_entropy / max(n_disp, 1), episode)
    writer.add_scalar("Stats/site_adv_mean", site_adv_t.mean().item(), episode)
    writer.add_scalar("Stats/site_ret_mean", site_ret_t.mean().item(), episode)


# ── Main training loop ─────────────────────────────────
def train():
    cfg = UAVHubConfig()
    set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training start [Option B: Site + Dispatch PPO]. Device: {device}")

    models_dir = os.path.join(current_module_dir, "models")
    logs_base_dir = os.path.join(current_module_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(logs_base_dir, f"run_{run_time}")
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_log_dir)

    env = RobustHubEnv(cfg)
    K = cfg.max_hubs

    policy = GraphAttentionPPO(N=env.N, node_dim=5, hidden_dim=128, K=K).to(device)

    initial_lr = 3e-4
    final_lr = 3e-5
    optimizer = Adam(policy.parameters(), lr=initial_lr)

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_clip_epsilon = 2.0
    ppo_epochs = 4

    update_timestep = 480
    max_episodes = 5000

    best_eval_cost = float('inf')
    last_eval_episode = 0
    eval_interval = 50

    initial_entropy_coef = 0.05
    final_entropy_coef = 0.001
    target_kl = 0.03

    episode = 0

    while episode < max_episodes:
        progress = episode / max_episodes
        anneal = max(0.0, (progress - 0.5) * 2.0)
        current_lr = initial_lr + (final_lr - initial_lr) * anneal
        entropy_coef = initial_entropy_coef + (final_entropy_coef - initial_entropy_coef) * anneal
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        trajectories = []
        collected_steps = 0

        while collected_steps < update_timestep and episode < max_episodes:
            episode += 1
            obs, _ = env.reset()
            traj = []

            # ── Site step ──
            t_obs = obs_to_tensors(obs, device)
            logits, value = policy(t_obs)
            hubs, site_lp = policy.sample_site(logits[0])
            next_obs, reward_site, _, _, info_site = env.step(np.array(hubs))

            traj.append({
                'obs':     clone_obs(t_obs),
                'action':  hubs,
                'log_prob': site_lp,
                'value':   value.squeeze().detach(),
                'reward':  reward_site,
            })

            ep_transport = 0.0
            ep_penalty = 0.0

            # ── Dispatch steps ──
            done = False
            lstm_hidden = None
            while not done:
                t_obs = obs_to_tensors(next_obs, device)
                t_obs['lstm_hidden'] = lstm_hidden

                logits, value, lstm_hidden = policy(t_obs)
                actions, per_node_lp, sum_lp = policy.sample_dispatch(logits)

                next_obs, reward, done, _, info = env.step(actions.cpu().numpy())

                # Per-node advantage = -per_node_cost / 100 (same scale as step reward)
                per_node_cost = torch.FloatTensor(info['per_node_cost']).to(device)
                per_node_adv = -per_node_cost / 100.0

                traj.append({
                    'obs':          clone_obs(t_obs),
                    'action':       actions.clone().detach(),
                    'log_prob':     sum_lp,
                    'per_node_lp':  per_node_lp.detach(),
                    'per_node_adv': per_node_adv,
                    'value':        value.squeeze().detach(),
                    'reward':       reward,
                })

                ep_transport += info['transport_cost']
                ep_penalty += info['unmet_penalty']

                if info['t'] in [32, 72]:
                    print(f"   [Dispatch] t={info['t']}/96 | active={info['active_orders']} | "
                          f"served={info['served_per_hub']}")

            collected_steps += len(traj)
            trajectories.append(traj)

            ep_cost = -reward_site * 100.0 + ep_transport + ep_penalty
            ep_cov = 1.0 - ep_penalty / max(ep_transport + ep_penalty, 1.0)

            writer.add_scalar("Train/Cost", ep_cost, episode)
            writer.add_scalar("Train/Coverage", ep_cov, episode)
            writer.add_scalar("Train/Transport", ep_transport, episode)
            writer.add_scalar("Train/Penalty", ep_penalty, episode)
            writer.add_scalar("Train/FixedCost", -reward_site * 100.0, episode)

            print(f"[Ep {episode:04d}] hubs={hubs} | cost={ep_cost:.0f} | "
                  f"transport={ep_transport:.0f} | penalty={ep_penalty:.0f} | cov={ep_cov*100:.1f}%")

        # ── PPO update ──
        ppo_update(
            trajectories, policy, optimizer, device,
            gamma, gae_lambda, clip_epsilon, value_clip_epsilon,
            ppo_epochs, target_kl, entropy_coef, writer, episode
        )
        writer.add_scalar("PPO/lr", current_lr, episode)

        # ── Eval ──
        if episode - last_eval_episode >= eval_interval:
            last_eval_episode = episode
            eval_env = RobustHubEnv(cfg, mode='eval')
            eval_info = deterministic_eval(eval_env, policy, device)
            eval_cost = eval_info['avg_cost']
            eval_cov = eval_info['avg_coverage']
            eval_hubs = eval_info['hub_locations']

            writer.add_scalar("Eval/Cost", eval_cost, episode)
            writer.add_scalar("Eval/Coverage", eval_cov, episode)

            if eval_cost < best_eval_cost and eval_cost > 0:
                best_eval_cost = eval_cost
                torch.save(policy.state_dict(), os.path.join(models_dir, "best_eval_policy.pth"))
                print(f"[EVAL BEST] Ep {episode:04d} | cost={eval_cost:.0f} | "
                      f"cov={eval_cov*100:.1f}% | hubs={eval_hubs}")
            else:
                print(f"[EVAL] Ep {episode:04d} | cost={eval_cost:.0f} | "
                      f"cov={eval_cov*100:.1f}% | hubs={eval_hubs}")

    final_model_path = os.path.join(models_dir, "final_gat_policy.pth")
    torch.save(policy.state_dict(), final_model_path)
    writer.close()
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()
