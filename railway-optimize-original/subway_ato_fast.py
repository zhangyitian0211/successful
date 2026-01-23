import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_utils
import matplotlib.pyplot as plt
import numpy as np
import os
from train_env_ato import train_env_ato

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return torch.tanh(x) * self.action_bound


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.device = device
        self.action_dim = action_dim

    def take_action(self, state):
        # 确保state是numpy数组
        if isinstance(state, np.ndarray):
            state = state.flatten()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim) * 1
        return action

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

    def save(self, filepath):
        """保存模型参数"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'sigma': self.sigma
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.sigma = checkpoint['sigma']
        print(f"模型已从 {filepath} 加载")

    def update(self, transition_dict):
        # 确保所有输入都是numpy数组
        states = np.array(transition_dict['states'], dtype=np.float32)
        actions = np.array(transition_dict['actions'], dtype=np.float32).reshape(-1, 1)
        rewards = np.array(transition_dict['rewards'], dtype=np.float32).reshape(-1, 1)
        next_states = np.array(transition_dict['next_states'], dtype=np.float32)
        dones = np.array(transition_dict['dones'], dtype=np.float32).reshape(-1, 1)

        # 转换为tensor
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        next_actions = self.actor_target(next_states)
        next_q_value = self.critic_target(next_states, next_actions)
        q_targets = rewards + self.gamma * next_q_value * (1 - dones)
        q_value = self.critic(states, actions)
        critic_loss = torch.mean(F.mse_loss(q_targets.detach(), q_value))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.soft_update(self.actor, self.actor_target)  # 软更新策略网络
        self.soft_update(self.critic, self.critic_target)  # 软更新价值网络


if __name__ == "__main__":
    # ========== 优化后的训练参数（适合CPU训练）==========
    actor_lr = 5e-4  # 增加Actor学习率
    critic_lr = 5e-3  # 增加Critic学习率
    num_episodes = 500  # 减少训练回合数：1000 -> 500
    hidden_dim = 64  # 增加隐藏层维度：32 -> 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 5000  # 减少缓冲区大小：10000 -> 5000
    minimal_size = 200  # 减少最小经验数：500 -> 200
    batch_size = 128  # 减少批处理大小：256 -> 128
    sigma = 0.05  # 降低探索噪声：0.1 -> 0.05
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")
    print("开始训练DDPG智能体（CPU优化版）...")
    print("=" * 50)
    print("参数设置:")
    print(f"  - 训练回合数: {num_episodes}")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print(f"  - Actor学习率: {actor_lr}")
    print(f"  - Critic学习率: {critic_lr}")
    print(f"  - 折扣因子: {gamma}")
    print(f"  - 软更新参数: {tau}")
    print(f"  - 探索噪声: {sigma}")
    print(f"  - 缓冲区大小: {buffer_size}")
    print(f"  - 批处理大小: {batch_size}")
    print(f"  - 更新频率: 4")
    print("=" * 50)

    env = train_env_ato()
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = 2
    action_dim = 1
    action_bound = 1  # 动作最大值 (最大加速度 1 m/s²)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, update_freq=2)
    episodes_list = list(range(len(return_list)))

    # 保存训练好的模型
    model_path = 'ddpg_model_ato_fast.pth'
    agent.save(model_path)

    # 绘制训练回报曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on subway ATO optimization (Fast CPU Version)')
    plt.grid(True, alpha=0.3)
    plt.savefig('ddpg_returns_ato_fast.png', dpi=300, bbox_inches='tight')
    print("\n训练回报曲线已保存为: ddp_returns_ato_fast.png")
    plt.close()

    print("\n开始测试训练好的智能体...")
    print("=" * 50)
    state, _ = env.reset()
    done = False
    truncated = False
    x_list = []
    v_list = []
    step_count = 0
    total_energy = 0

    # 测试时关闭噪声
    original_sigma = agent.sigma
    agent.sigma = 0

    while not (done or truncated):
        action = agent.take_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        x, v = state
        x_list.append(x)
        v_list.append(v)
        total_energy = float(env.P)
        step_count += 1

    # 恢复噪声
    agent.sigma = original_sigma

    print(f"测试完成，共执行 {step_count} 步")
    if truncated:
        print("警告: 测试超时，智能体未能在规定步数内到达终点")
    print(f"最终位置: {x:.2f} m / {env.total_distance} m")
    print(f"最终速度: {v:.2f} m/s ({v*3.6:.2f} km/h)")
    print(f"总能耗: {total_energy:.2f} kW")
    print(f"平均速度: {env.total_distance / step_count:.2f} m/s ({env.total_distance / step_count * 3.6:.2f} km/h)")
    print("=" * 50)

    # 绘制位置-速度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x_list, v_list, linewidth=2)
    plt.xlabel('Position x (m)', fontsize=12)
    plt.ylabel('Velocity v (m/s)', fontsize=12)
    plt.title('ATO Optimization: Position-Velocity Curve (Fast CPU Version)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 标注坡度变化点（更新为所有8个点）
    gradient_points = [161, 361, 691, 1351, 1881, 2251, 2451]
    for point in gradient_points:
        plt.axvline(x=point, color='r', linestyle='--', alpha=0.5, linewidth=1)

    plt.legend(['v-x curve', 'Gradient change points'], loc='upper right')
    try:
        plt.savefig('x_v_curve_ato_fast.png', dpi=300, bbox_inches='tight')
        print("位置-速度曲线已保存为: x_v_curve_ato_fast.png")
    except Exception as e:
        print(f"保存x_v曲线时出错: {e}")
    finally:
        plt.close()

    print("\n所有任务完成！")
