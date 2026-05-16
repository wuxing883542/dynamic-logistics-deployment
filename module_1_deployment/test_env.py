import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import UAVHubConfig
from module_1_deployment.env_robust_hub import RobustHubEnv

def test_random_agent():
    cfg = UAVHubConfig()
    
    # 【请确保你在 config.py 中把 Q 改小了，比如 Q=400】
    print(f"🔧 当前枢纽容量 Q = {cfg.Q}, 拒单惩罚系数 = {cfg.penalty_unmet}")
    
    env = RobustHubEnv(cfg)
    K = cfg.max_hubs
    N = env.N
    
    print("🚀 开始环境测试 (Random Agent)...\n")
    
    for episode in range(3): # 测试 3 个完整的日子
        obs, _ = env.reset()
        print(f"📅 [Episode {episode+1}] 开启全新的一天!")
        
        # 1. 选址动作 (随机选 K 个不重复的节点)
        random_hubs = np.random.choice(N, K, replace=False)
        obs, reward_site, _, _, info_site = env.step(random_hubs)
        print(f"   ➤ [选址] 建立枢纽: {random_hubs}, 建站惩罚(缩放后): {reward_site:.2f}")
        
        total_transport = 0.0
        total_penalty = 0.0
        done = False
        
        # 2. 分配动作 (模拟 96 个时间槽)
        while not done:
            current_orders = obs['current_orders']
            # 随机生成一个分配矩阵: 0,1,2 是枢纽，3 是拒单
            random_dispatch = np.random.randint(0, K + 1, size=N)
            
            obs, reward_dispatch, done, _, info = env.step(random_dispatch)
            
            total_transport += info['transport_cost']
            total_penalty += info['unmet_penalty']
            
            # 在早晚高峰打印一下状态
            if info['t'] in [32, 72]: # 对应早上8点和晚上18点左右
                print(f"   ➤ [分配] 时间槽 {info['t']}/96 | 活跃订单点: {info['active_orders']} | "
                      f"各枢纽已服务: {info['served_per_hub']}")
                
        total_daily_cost = total_transport + total_penalty
        print(f"   🏁 [结算] 当日运输总成本: {total_transport:.1f}, 拒单总惩罚: {total_penalty:.1f}")
        print(f"   🏁 [结算] 瞎选策略导致当日总成本: {total_daily_cost:.1f}\n")

if __name__ == "__main__":
    test_random_agent()