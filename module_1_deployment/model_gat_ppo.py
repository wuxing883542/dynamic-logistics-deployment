import torch
import torch.nn as nn
from torch.distributions import Categorical


class GraphAttentionPPO(nn.Module):
    """Option B: Site-Dispatch PPO with shared Transformer backbone.

    Phase 0 (site selection): Plackett-Luce sequential sampling → K unique hubs
    Phase 1 (dispatch): N independent Categorical(K+1) → per-node hub/reject
    """

    def __init__(self, N=98, node_dim=5, hidden_dim=128, K=3):
        super().__init__()
        self.N = N
        self.K = K
        self.hidden_dim = hidden_dim

        # ── Shared backbone ──
        self.node_embed = nn.Linear(node_dim + 1, hidden_dim)  # +1 for hub_mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Phase 1: Site head ──
        self.site_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # ── Phase 2: Dispatch head ──
        # Policy LSTM: temporal demand tracking for dispatch decisions
        self.policy_lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.order_proj = nn.Linear(hidden_dim + 1 + 64, 64)
        self.hub_proj = nn.Linear(hidden_dim + 1, 64)
        self.reject_feat = nn.Parameter(torch.randn(1, 1, 64))

        # ── Critic (site / dispatch 分离; dispatch Critic 无 LSTM，避免梯度冲突) ──
        self.site_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # dispatch Critic: node stats + demand stats (no LSTM — stateless, stable)
        self.dispatch_critic = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    # ── Shared encoding ────────────────────────────────
    def _encode_nodes(self, node_features, hub_mask):
        x = torch.cat([node_features, hub_mask.unsqueeze(-1)], dim=-1)
        x = self.node_embed(x)
        return self.transformer(x)

    # ── Unified forward ────────────────────────────────
    def forward(self, obs_batch):
        """obs_batch keys:
        - phase: int (0=site, 1=dispatch)
        - node_features: (B, N, 5)
        - hub_mask: (B, N)
        - current_orders: (B, N)        [dispatch only]
        - hub_capacities: (B, K)        [dispatch only]
        - lstm_hidden: optional tuple   [dispatch only]
        Returns:
        - phase=0: (logits (B,N), value (B,))
        - phase=1: (logits (B,N,K+1), value (B,), lstm_hidden)
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
            return self._forward_dispatch(node_emb, orders, hub_caps, hub_m, lstm_hidden)

    # ── Phase 0: Site selection ────────────────────────
    def _forward_site(self, node_emb):
        logits = self.site_scorer(node_emb).squeeze(-1)
        # Critic on detached emb — prevents Critic gradient from distorting shared Transformer
        val_feat = torch.cat([node_emb.detach().mean(dim=1), node_emb.detach().max(dim=1)[0]], dim=-1)
        value = self.site_critic(val_feat).squeeze(-1)
        return logits, value

    # ── Phase 1: Dispatch ──────────────────────────────
    def _forward_dispatch(self, node_emb, orders, hub_caps, hub_mask, lstm_hidden):
        B = orders.size(0)

        # Policy LSTM: temporal demand stats
        total_demand = orders.sum(dim=-1, keepdim=True)
        demand_std = orders.std(dim=-1, keepdim=True)
        demand_max = orders.max(dim=-1)[0].unsqueeze(-1)
        lstm_input = torch.cat([total_demand, demand_std, demand_max], dim=-1).unsqueeze(1)

        if lstm_hidden is None:
            lstm_out, lstm_hidden = self.policy_lstm(lstm_input)
        else:
            lstm_out, lstm_hidden = self.policy_lstm(lstm_input, lstm_hidden)

        # Order queries: node_emb + order_qty + LSTM context
        orders_expanded = orders.unsqueeze(-1)
        lstm_expanded = lstm_out.expand(-1, self.N, -1)
        q = self.order_proj(torch.cat([node_emb, orders_expanded, lstm_expanded], dim=-1))

        # Hub keys: extract hub node embeddings
        hub_embs = self._extract_hub_embeddings(node_emb, hub_mask)
        cap_expanded = hub_caps.unsqueeze(-1)
        k_hubs = self.hub_proj(torch.cat([hub_embs, cap_expanded], dim=-1))

        # Append reject key
        k_reject = self.reject_feat.expand(B, 1, -1)
        k_all = torch.cat([k_hubs, k_reject], dim=1)

        # Cross-attention
        logits = torch.bmm(q, k_all.transpose(1, 2)) / (64 ** 0.5)

        # Dispatch Critic: detached emb + demand stats — no gradient into Transformer
        emb_d = node_emb.detach()
        val_feat = torch.cat([
            emb_d.mean(dim=1), emb_d.max(dim=1)[0],
            total_demand, demand_std, demand_max,
        ], dim=-1)
        value = self.dispatch_critic(val_feat).squeeze(-1)

        return logits, value, lstm_hidden

    def _extract_hub_embeddings(self, node_emb, hub_mask):
        """Extract K hub embeddings from node_emb using hub_mask. Returns (B, K, H)."""
        B = node_emb.size(0)
        hub_mask_bool = hub_mask > 0.5
        hub_embs = []
        for b in range(B):
            indices = torch.where(hub_mask_bool[b])[0]
            hub_embs.append(node_emb[b, indices])
        return torch.stack(hub_embs, dim=0)

    # ── Sampling helpers ───────────────────────────────
    def sample_site(self, logits, deterministic=False):
        """Plackett-Luce sequential sampling without replacement.

        Returns (indices list of K ints, log_prob scalar).
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
        """Per-node independent Categorical over K+1 options.

        Returns (actions (N,), per_node_lp (N,), sum_lp scalar).
        """
        dist = Categorical(logits=logits.squeeze(0))
        if deterministic:
            actions = logits.squeeze(0).argmax(dim=-1)
        else:
            actions = dist.sample()
        per_node_lp = dist.log_prob(actions)  # (N,)
        return actions, per_node_lp, per_node_lp.sum()

    def compute_site_log_prob(self, logits, hubs):
        """Recompute Plackett-Luce log_prob for a given hub sequence."""
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
        """Recompute per-node log_probs. Returns (per_node (N,), sum)."""
        dist = Categorical(logits=logits.squeeze(0))
        per_node = dist.log_prob(actions.squeeze(0))  # (N,)
        return per_node, per_node.sum()
