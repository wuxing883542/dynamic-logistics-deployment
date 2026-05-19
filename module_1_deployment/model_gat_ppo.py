import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv

# =========================================================================
# 🎮 大脑一：主调度策略网络 (DynamicDispatchPPO) - 纯空间决策大脑
# =========================================================================
class DynamicDispatchPPO(nn.Module):
    """
    纯血空间分配大脑 (GAT + MLP)
    💡 IPPO 更新：Critic 不再输出全局标量，而是为每个节点输出独立的 Value (B, N)
    💡 CTDE 更新：Critic 额外接收全局宏观特征（上帝视角），Actor 仅依赖局部特征
    """
    # 💡 注意：这里的 node_feature_dim 默认值已经改成了 9
    def __init__(self, cfg, N, node_feature_dim=9, hidden_dim=128):
        super(DynamicDispatchPPO, self).__init__()
        self.K = cfg.max_hubs
        self.N = N
        self.hidden_dim = hidden_dim

        # --- 0. 预构建静态全连接拓扑图缓冲区 ---
        row = torch.arange(N).repeat_interleave(N)
        col = torch.arange(N).repeat(N)
        self.register_buffer('base_edge_index', torch.stack([row, col], dim=0))

        # --- 1. 空间特征提取层 (GAT) ---
        self.gat1 = GATConv(node_feature_dim, hidden_dim // 2, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        # --- 2. 决策层 (MLP) ---
        # Actor: 负责为每个节点挑选枢纽或拒单
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + self.K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K + 1)
        )
        
        # Critic: 💡 改为评估每个节点的局部分配价值
        # 接收 actor_input (hidden_dim + K) + 全局宏观特征 (3维)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + self.K + 3, hidden_dim), # 👈 增加了 3 维
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _get_batched_edge_index(self, batch_size):
        offset = (torch.arange(batch_size, device=self.base_edge_index.device) * self.N).view(-1, 1, 1)
        return (self.base_edge_index.unsqueeze(0) + offset).transpose(1, 2).reshape(2, -1)

    def forward(self, obs, action_mask=None):
        node_features = obs['node_features']       
        current_orders = obs['current_orders']     
        hub_mask = obs['hub_mask']                 
        hub_capacities = obs['hub_capacities']     
        predicted_orders = obs['predicted_orders'] 

        original_dim = node_features.dim()
        if original_dim == 2:
            node_features = node_features.unsqueeze(0)
            current_orders = current_orders.unsqueeze(0)
            hub_mask = hub_mask.unsqueeze(0)
            hub_capacities = hub_capacities.unsqueeze(0)
            predicted_orders = predicted_orders.unsqueeze(0)
            if action_mask is not None: action_mask = action_mask.unsqueeze(0)

        B, N_dim, _ = node_features.shape

        x_node = torch.cat([
            node_features, 
            current_orders.unsqueeze(-1), 
            hub_mask.unsqueeze(-1),
            predicted_orders.unsqueeze(-1)
        ], dim=-1)

        x_node_flat = x_node.view(B * self.N, -1)
        batched_edge_index = self._get_batched_edge_index(B)

        x = F.elu(self.gat1(x_node_flat, batched_edge_index))
        node_embeddings = F.elu(self.gat2(x, batched_edge_index)).view(B, self.N, self.hidden_dim)
        
        hub_cap_expanded = hub_capacities.unsqueeze(1).expand(-1, self.N, -1)
        actor_input = torch.cat([node_embeddings, hub_cap_expanded], dim=-1) # (B, N, hidden_dim + K)
        
        logits = self.actor_head(actor_input)

        # 💡 [CTDE 核心修改] 为 Critic 组装“上帝视角”的宏观统计量
        # 1. 剩余总运力比率
        total_rem_cap = hub_capacities.sum(dim=-1, keepdim=True) / self.K  # (B, 1)
        # 2. 当前步全城总需求 (除以100控制量级)
        total_demand = current_orders.sum(dim=-1, keepdim=True) / 100.0    # (B, 1) 
        # 3. 活跃节点数量占比
        active_ratio = (current_orders > 0).float().sum(dim=-1, keepdim=True) / self.N # (B, 1)

        global_context = torch.cat([total_rem_cap, total_demand, active_ratio], dim=-1) # (B, 3)
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.N, -1)    # (B, N, 3)

        # 💡 Critic 专属输入 = 局部特征 + 上帝视角
        critic_input = torch.cat([actor_input, global_context_expanded], dim=-1) # (B, N, hidden_dim + K + 3)
        
        value = self.critic_head(critic_input).squeeze(-1) # 形状: (B, N)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.to(torch.bool), -1e9)

        if original_dim == 2:
            logits = logits.squeeze(0)
            value = value.squeeze(0) # 降维后形状: (N,)

        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, obs, action_mask=None, deterministic=False):
        dist, value = self.forward(obs, action_mask)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        
        # 💡 [IPPO核心修改] 绝对不能 sum！保留每个节点的独立 log_prob
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# =========================================================================
# 🔮 大脑二：辅助预测网络 (FutureDemandPredictor) - 无修改
# =========================================================================
class FutureDemandPredictor(nn.Module):
    def __init__(self, N, history_len=12, pred_len=4, hidden_dim=64):
        super().__init__()
        self.N = N
        self.history_len = history_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
                
        self.lstm = nn.LSTM(input_size=N, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * N)
        )

    def forward(self, hist_orders):
        original_dim = hist_orders.dim()
        if original_dim == 2:
            hist_orders = hist_orders.unsqueeze(0) 

        lstm_out, _ = self.lstm(hist_orders) 
        last_hidden = lstm_out[:, -1, :]      
        pred_flat = self.fc(last_hidden)      
        pred = pred_flat.view(-1, self.pred_len, self.N)
        
        if original_dim == 2:
            pred = pred.squeeze(0) 

        return pred