import torch
import torch.nn as nn
from torch.distributions import Categorical

class GraphAttentionPPO(nn.Module):
    """
    基于空间偏置图注意力网络 (Spatial-Aware GAT) 的 PPO 决策大脑
    输入: 城市节点矩阵 (N, 3) -> [x坐标, y坐标, 是否为枢纽]
    输出: 动作概率分布 (Discrete) & 当前状态价值 (Value)
    """
    def __init__(self, N=20, max_hubs=5, hidden_dim=128):
        super(GraphAttentionPPO, self).__init__()
        self.N = N
        self.max_hubs = max_hubs
        self.action_dim = max_hubs * (N + 1)
        
        # 1. 节点特征嵌入层
        self.node_embed = nn.Linear(3, hidden_dim)
        
        # 💡 [新增]: 空间图拓扑注入器 (Spatial Edge Projector)
        # 作用: 将物理距离转化为图的“边权重惩罚”，注入到注意力机制中
        self.num_heads = 4
        self.spatial_bias_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_heads) # 为每个 Attention Head 学习一个独特的距离衰减策略
        )
        
        # 2. 图注意力层 (将标准 Transformer 升级为 GAT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=self.num_heads,            
            dim_feedforward=256,
            batch_first=True
        )
        self.gat_layers = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. PPO 决策头
        flatten_dim = N * hidden_dim
        
        self.actor_net = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        self.critic_net = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        
        # ==================================================
        # 🕸️ [阶段一：动态图结构重构 (Graph Construction)]
        # ==================================================
        # 1. 提取物理坐标 -> (Batch, N, 2)
        coords = obs[:, :, :2] 
        
        # 2. 利用广播机制，动态计算节点间的物理距离矩阵 -> (Batch, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1) # (Batch, N, N, 2)
        dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-5)          # (Batch, N, N)
        
        # 3. 将距离标量投影为多头注意力偏置 -> (Batch, N, N, Heads)
        dist_features = dist_matrix.unsqueeze(-1)        # (Batch, N, N, 1)
        spatial_bias = self.spatial_bias_proj(dist_features)
        
        # 4. 调整形状以匹配 PyTorch 的 src_mask 要求: (Batch * Heads, N, N)
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)  # (Batch, Heads, N, N)

        # 取负绝对值作为距离惩罚，clamp 到 [-5, 0] 防止数值过大导致 softmax 输出 NaN
        spatial_mask = -torch.abs(spatial_bias)           # (Batch, Heads, N, N)
        spatial_mask = torch.clamp(spatial_mask, min=-5.0)
        spatial_mask = spatial_mask.reshape(batch_size * self.num_heads, self.N, self.N)
        
        # ==================================================
        # 🧠 [阶段二：带有空间拓扑的消息传递 (Message Passing)]
        # ==================================================
        # 5. 节点嵌入 -> (Batch, N, 128)
        x = self.node_embed(obs)
        
        # 6. 传入 spatial_mask！PyTorch 会将其直接加到 Attention Logits 上。
        # 此时的 Transformer 已经变成了融合了物理距离的纯正 Graph Attention Network!
        graph_out = self.gat_layers(x, mask=spatial_mask)
        
        # ==================================================
        # 🎯 [阶段三：动作与价值输出]
        # ==================================================
        # 7. 展平整个城市的特征图 -> (Batch, N * 128)
        flat_out = graph_out.reshape(batch_size, -1)
        
        # 8. 计算 Critic 价值与 Actor Logits
        value = self.critic_net(flat_out)
        logits = self.actor_net(flat_out)
        
        dist = Categorical(logits=logits)
        
        return dist, value

    def act(self, obs):
        device = next(self.parameters()).device 
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, value = self(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs, action):
        dist, value = self(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy