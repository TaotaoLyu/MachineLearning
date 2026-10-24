import gymnasium as gym
import torch
import numpy as np
import random

seed=53
def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

env=gym.make("LunarLander-v3", render_mode="rgb_array")
fix(env, seed)

print(env.action_space)
initial_state=env.reset()
print(initial_state)
random_action=env.action_space.sample()
print(random_action)
observation, reward, terminated, truncated, info=env.step(random_action)
done = terminated or truncated
print(done)
print(info)
print(reward)

from pathlib import Path
from PIL import Image
img = env.render()
Image.fromarray(img).save(Path(__file__).with_name("lunar_lander.png"))
# env.close()




import torch.nn as nn
import torch.nn.functional as F
class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(8, 16)
        self.fc2=nn.Linear(16, 16)
        self.fc3=nn.Linear(16, 4)

    def forward(self, x):
        hid=torch.tanh(self.fc1(x))
        hid=torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

import torch.optim as optim
from torch.distributions import Categorical
class PolicyGradientAgent():
    def __init__(self, network):
        self.network=network
        self.optimizer=optim.SGD(self.network.parameters(), lr=0.001)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss=(-log_probs*rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def sample(self, state):
        action_prob=self.network(torch.FloatTensor(state))
        action_dist=Categorical(action_prob)
        action=action_dist.sample()
        log_prob=action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, PATH):
        Agent_Dict={
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH):
        checkpoint=torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

network=PolicyGradientNetwork()
agent=PolicyGradientAgent(network)

from tqdm import tqdm
import matplotlib.pyplot as plt
def Train():
    agent.network.train()
    EPISODE_PER_BATCH=5
    NUM_BATCH=400
    gamma=0.99

    avg_total_rewards, avg_final_rewards=[], []

    prg_bar=tqdm(range(NUM_BATCH), dynamic_ncols=True)
    for batch in prg_bar:
        log_probs, rewards=[], []
        total_rewards, final_rewards=[], []

        for episode in range(EPISODE_PER_BATCH):

            state,_=env.reset()
            total_reward, total_step=0, 0

            seq_rewards=[]
            while True:
                action, log_prob=agent.sample(state)
                next_state, reward, terminated, truncated, info=env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                state=next_state
                total_reward+=reward
                total_step+=1
                seq_rewards.append(reward)

                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    break

            discounted_return=0.0
            seq_returns=[]
            for r in reversed(seq_rewards):
                discounted_return=r+gamma*discounted_return
                seq_returns.append(discounted_return)
            rewards.extend(reversed(seq_returns))


        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")


        rewards=(rewards-np.mean(rewards))/(np.std(rewards)+1e-9)
        log_probs_tensor=torch.stack(log_probs)
        rewards_tensor=torch.from_numpy(rewards)
        agent.learn(log_probs_tensor, rewards_tensor)
        prg_bar.set_postfix(
            rewards=tuple(rewards_tensor.shape),
            log_probs=tuple(log_probs_tensor.shape),
        )

    plt.figure()
    plt.plot(avg_total_rewards, label="Total Reward")
    plt.title("Average Total Reward per Batch")
    plt.savefig(Path(__file__).with_name("avg_total_rewards.png"))
    plt.close()

    plt.figure()
    plt.plot(avg_final_rewards, label="Final Reward")
    plt.title("Average Final Reward per Batch")
    plt.savefig(Path(__file__).with_name("avg_final_rewards.png"))
    agent.save(Path(__file__).with_name("policy_gradient_agent.pth"))
    plt.close()

Train()
