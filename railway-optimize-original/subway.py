import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_utils
import matplotlib.pyplot as plt
import numpy as np
from train_env_final import train_env1

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

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

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
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 2000
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 1000
    minimal_size = 100
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"使用设备: {device}")
    print("开始训练DDPG智能体...")

    env = train_env1()
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = 2
    action_dim = 1
    action_bound = 1  # 动作最大值
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    episodes_list = list(range(len(return_list)))

    # 绘制训练回报曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on subway_optimize')
    plt.grid(True, alpha=0.3)
    plt.savefig('ddpg_returns.png', dpi=300, bbox_inches='tight')
    print("训练回报曲线已保存为: ddp_returns.png")
    plt.show()

    print("\n开始测试训练好的智能体...")
    state, _ = env.reset()
    done = False
    truncated = False
    x_list = []
    v_list = []
    step_count = 0
    while not (done or truncated):
        action = agent.take_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        x, v = state
        x_list.append(x)
        v_list.append(v)
        step_count += 1

    print(f"测试完成，共执行 {step_count} 步")
    print(f"最终位置: {x:.2f} m, 最终速度: {v:.2f} m/s")

    # 绘制位置-速度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x_list, v_list)
    plt.xlabel('Position x (m)')
    plt.ylabel('Velocity v (m/s)')
    plt.title('x-v figure')
    plt.grid(True, alpha=0.3)
    plt.savefig('x_v_curve.png', dpi=300, bbox_inches='tight')
    print("位置-速度曲线已保存为: x_v_curve.png")
    plt.show()

