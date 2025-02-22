import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


#############################################
# 定义 Actor 网络
#############################################
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob

#############################################
# 定义 Critic 网络（用于奖励值估计）
#############################################
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

#############################################
# 定义 SafeCritic 网络（用于安全代价估计）
#############################################
class SafeCritic(nn.Module):
    def __init__(self, args):
        super(SafeCritic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh] # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (SafeCritic)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        cost_value = self.fc3(s)
        return cost_value

class PPO_discrete:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # 新增安全相关参数
        self.cost_gamma = args.gamma  # 安全代价折扣因子（与critic相同）
        self.cost_lamda = args.lamda  # 安全代价 GAE 参数（与critic相同）
        self.cost_limit = args.cost_limit  # 安全代价阈值
        self.lr_multiplier = args.lr_multiplier  # 拉格朗日乘子更新学习率

        self.actor = Actor(args)
        self.critic = Critic(args)

        # 新增safe_critic网络
        self.safe_critic = SafeCritic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_safe_critic = torch.optim.Adam(self.safe_critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_safe_critic = torch.optim.Adam(self.safe_critic.parameters(), lr=self.lr_c)

        # 初始化拉格朗日乘子（确保非负）
        self.lambda_cost = torch.tensor(1.0, requires_grad=True)
        self.optimizer_lambda = torch.optim.Adam([self.lambda_cost], lr=self.lr_multiplier)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))  # 创建离散概率分布
            a = dist.sample()   # 从该分布中采样一个动作
            a_logprob = dist.log_prob(a)  # 计算该动作的对数概率 log π(a|s)
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self, replay_buffer, total_steps):
        # replay_buffer 已经修改为返回安全代价 c
        s, a, a_logprob, r, s_, dw, done, c = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs   # 计算TD误差
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs  # 计算状态值目标
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # 计算安全代价优势 cost_adv 和安全目标值 cost_target（使用 GAE）
        cost_adv = []
        gae_c = 0
        with torch.no_grad():
            cost_values = self.safe_critic(s)
            cost_values_next = self.safe_critic(s_)
            cost_deltas = c + self.cost_gamma * (1.0 - dw) * cost_values_next - cost_values
            for delta, d in zip(reversed(cost_deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae_c = delta + self.cost_gamma * self.cost_lamda * gae_c * (1.0 - d)
                cost_adv.insert(0, gae_c)
            cost_adv = torch.tensor(cost_adv, dtype=torch.float).view(-1, 1)
            cost_target = cost_adv + cost_values
            if self.use_adv_norm:  # Trick 1:advantage normalization
                cost_adv = ((cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

                # 安全部分（这里不进行 clip，直接乘以 cost_adv）
                actor_loss_cost = ratios * cost_adv[index]

                # 加入安全部分后的actor损失
                actor_loss = actor_loss + self.lambda_cost * actor_loss_cost

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                # 更新安全 critic（安全代价值函数）的损失
                v_cost = self.safe_critic(s[index])
                safe_critic_loss = F.mse_loss(cost_target[index], v_cost)
                self.optimizer_safe_critic.zero_grad()
                safe_critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.safe_critic.parameters(), 0.5)
                self.optimizer_safe_critic.step()

        # 更新拉格朗日乘子 lambda_cost
        # 计算整个 batch 上的平均安全代价优势与安全阈值的偏差
        cost_violation = cost_adv.mean() - self.cost_limit
        lambda_loss = - self.lambda_cost * cost_violation.detach()  # 注意 detach，避免影响其他梯度
        self.optimizer_lambda.zero_grad()
        lambda_loss.backward()
        self.optimizer_lambda.step()
        # 保证 lambda_cost 非负
        with torch.no_grad():
            self.lambda_cost.clamp_(0)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        # 加入safe_critic 和 拉格朗日乘子 的学习率衰减
        for p in self.optimizer_safe_critic.param_groups:
            p['lr'] = lr_c_now
        for p in self.optimizer_lambda.param_groups:
            p['lr'] = self.lr_multiplier * (1 - total_steps / self.max_train_steps)

    def save_model(self, total_steps=0, filename_prefix='ppo_model'):
        """Save the model, optimizer states, and total_steps to a file with a unique filename."""

        # 创建一个基于当前时间戳和训练步骤的唯一文件名
        filename = f"{filename_prefix}_{total_steps}.pth"

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'total_steps': total_steps,  # 保存 total_steps
        }, filename)

    # def load_model(self, total_steps=0, filename_prefix='ppo_model'):
    #     """Load the model, optimizer states and total_steps from a file."""
    #     filename = f"{filename_prefix}_{total_steps}.pth"
    #
    #     checkpoint = torch.load(filename)
    #     self.actor.load_state_dict(checkpoint['actor_state_dict'])
    #     self.critic.load_state_dict(checkpoint['critic_state_dict'])
    #     self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
    #     self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
    #     total_steps = checkpoint['total_steps']  # 恢复 total_steps
    #     print(f"Model and training state loaded from {filename}")
