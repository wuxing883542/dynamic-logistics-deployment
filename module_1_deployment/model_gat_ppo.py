import torch
import torch.nn as nn
from torch.distributions import Categorical

class GraphAttentionPPO(nn.Module):
    """方案 B: 带有共享 Transformer 骨干网络的 选址-调度 PPO。

    阶段 0 (选址): Plackett-Luce 序列无放回采样 → 选出 K 个不重复的枢纽
    阶段 1 (调度): N 个节点独立的 Categorical(K+1) 分类分布 → 每个节点选择枢纽或拒单
    """

    def __init__(self, N=98, node_dim=5, hidden_dim=128, K=3):
        super().__init__()
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # ── 共享骨干网络 (Shared backbone) ──
        self.node_embed = nn.Linear(node_dim + 1, hidden_dim)  # +1 是为了拼接 hub_mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── 阶段 1: 选址头 (Site head) ──
        self.site_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # ── 阶段 2: 调度头 (Dispatch head) ──
        # 策略 LSTM: 用于跟踪调度决策中的时间维度需求情绪
        self.policy_lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.order_proj = nn.Linear(hidden_dim + 1 + 64, 64)
        self.hub_proj = nn.Linear(hidden_dim + 1, 64)
        self.reject_feat = nn.Parameter(torch.randn(1, 1, 64))

        # ── 价值评估网络 (Critic: 仅保留选址 Critic，彻底删除无用的调度 Critic) ──
        self.site_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    # ── 共享特征编码 ────────────────────────────────
    def _encode_nodes(self, node_features, hub_mask):
        x = torch.cat([node_features, hub_mask.unsqueeze(-1)], dim=-1)
        x = self.node_embed(x)
        return self.transformer(x)

    # ── 统一前向传播 ────────────────────────────────
    def forward(self, obs_batch, detach_backbone=False):
        """obs_batch 字典键值:
        - phase: 整数 (0=选址, 1=调度)
        ... [其余省略]
        返回:
        - phase=0: (打分 logits (B,N), 价值 value (B,))
        - phase=1: (打分 logits (B,N,K+1), lstm_hidden) 
        """
        node_f = obs_batch['node_features']
        hub_m = obs_batch['hub_mask']
        node_emb = self._encode_nodes(node_f, hub_m)

        phase = obs_batch['phase']
        if isinstance(phase, torch.Tensor):
            phase = phase.item() if phase.numel() == 1 else phase[0].item()

        if phase == 0:
            return self._forward_site(node_emb)
        else:
            orders = obs_batch['current_orders']
            hub_caps = obs_batch['hub_capacities']
            lstm_hidden = obs_batch.get('lstm_hidden', None)
            
            # 💡 [核心修复] 提取严格对齐的枢纽序列位置 (Shape: B, K)
            hub_locations = obs_batch['hub_locations'] 
            
            return self._forward_dispatch(node_emb, orders, hub_caps, hub_m, hub_locations, lstm_hidden, detach_backbone)

    # ── 阶段 0: 选址决策 ────────────────────────
    def _forward_site(self, node_emb):
        logits = self.site_scorer(node_emb).squeeze(-1)
        # 在截断梯度的特征上进行 Critic 评估 —— 防止 Critic 的梯度扭曲共享的 Transformer
        val_feat = torch.cat([node_emb.detach().mean(dim=1), node_emb.detach().max(dim=1)[0]], dim=-1)
        value = self.site_critic(val_feat).squeeze(-1)
        return logits, value

    # ── 阶段 1: 调度分配 ──────────────────────────────
    def _forward_dispatch(self, node_emb, orders, hub_caps, hub_mask, hub_locations, lstm_hidden, detach_backbone):
        B = orders.size(0)

        if detach_backbone:
            node_emb_use = node_emb.detach()
        else:
            node_emb_use = node_emb

        # 策略 LSTM: 时间维度的需求统计
        total_demand = orders.sum(dim=-1, keepdim=True)
        demand_std = orders.std(dim=-1, keepdim=True)
        demand_max = orders.max(dim=-1)[0].unsqueeze(-1)
        lstm_input = torch.cat([total_demand, demand_std, demand_max], dim=-1).unsqueeze(1)

        if lstm_hidden is None:
            lstm_out, lstm_hidden = self.policy_lstm(lstm_input)
        else:
            lstm_out, lstm_hidden = self.policy_lstm(lstm_input, lstm_hidden)

        # 订单查询向量 (Query): 节点特征 + 订单量 + LSTM 宏观上下文
        orders_expanded = orders.unsqueeze(-1)
        lstm_expanded = lstm_out.expand(-1, self.N, -1)
        
        q = self.order_proj(torch.cat([node_emb_use, orders_expanded, lstm_expanded], dim=-1))

        # ── 枢纽键向量 (Key) 提取 ──
        # 💡 [核心修复] 利用批量高级索引，严格按照 Plackett-Luce 的采样物理顺序从 Transformer 编码中提取特征！
        batch_idx = torch.arange(B, device=node_emb_use.device).unsqueeze(-1) # (B, 1)
        hub_embs = node_emb_use[batch_idx, hub_locations]                     # (B, K, hidden_dim)

        cap_expanded = hub_caps.unsqueeze(-1)
        k_hubs = self.hub_proj(torch.cat([hub_embs, cap_expanded], dim=-1))

        # 拼接“拒单”的虚拟键向量
        k_reject = self.reject_feat.expand(B, 1, -1)
        k_all = torch.cat([k_hubs, k_reject], dim=1)

        # 交叉注意力计算 (Cross-attention)
        logits = torch.bmm(q, k_all.transpose(1, 2)) / (64 ** 0.5)

        return logits, lstm_hidden

    # ── 采样辅助函数 ───────────────────────────────
    def sample_site(self, logits, deterministic=False):
        """Plackett-Luce 无放回序列采样。
        返回 (包含 K 个整数索引的列表, 对数概率标量)。
        """
        hubs = []
        log_prob = 0.0
        remaining = torch.ones(self.N, dtype=torch.bool, device=logits.device)

        for _ in range(self.K):
            masked = logits.masked_fill(~remaining, -1e9)
            dist = Categorical(logits=masked)
            if deterministic:
                h = masked.argmax()
            else:
                h = dist.sample()
            log_prob += dist.log_prob(h)
            hubs.append(h.item() if h.numel() == 1 else h.cpu().item())
            remaining[h] = False

        return hubs, log_prob

    def sample_dispatch(self, logits, deterministic=False):
        """为每个节点在 K+1 个选项中做独立的 Categorical 采样。
        返回 (动作 (N,), 每个节点的对数概率 (N,), 总对数概率标量)。
        """
        dist = Categorical(logits=logits.squeeze(0))
        if deterministic:
            actions = logits.squeeze(0).argmax(dim=-1)
        else:
            actions = dist.sample()
        per_node_lp = dist.log_prob(actions)  # (N,)
        return actions, per_node_lp, per_node_lp.sum()

    def compute_site_log_prob(self, logits, hubs):
        """为给定的枢纽序列重新计算 Plackett-Luce 对数概率。"""
        log_prob = 0.0
        remaining = torch.ones(self.N, dtype=torch.bool, device=logits.device)
        for h in hubs:
            masked = logits.masked_fill(~remaining, -1e9)
            dist = Categorical(logits=masked)
            h_t = torch.tensor(h, device=logits.device, dtype=torch.long)
            log_prob += dist.log_prob(h_t)
            remaining[h] = False
        return log_prob

    def compute_dispatch_log_prob(self, logits, actions):
        """重新计算每个节点的对数概率。返回 (每个节点的对数概率 (N,), 总和)。"""
        dist = Categorical(logits=logits.squeeze(0))
        per_node = dist.log_prob(actions.squeeze(0))  # (N,)
        return per_node, per_node.sum()