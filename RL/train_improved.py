import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Categorical
from tqdm import tqdm

SEED = 53
NUM_BATCH = 2000
EPISODE_PER_BATCH = 5
GAMMA = 0.999
LEARNING_RATE = 0.0001
VALUE_LEARNING_RATE = 0.001
VALUE_UPDATES = 20
ENTROPY_COEF = 0.001
MAX_EPISODE_STEPS = 1000
EVAL_INTERVAL = 50
DEVICE = "cpu"  # 可改为 "cuda"；Box2D 环境本身仍在 CPU 上运行。


def fix(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)
    env.action_space.seed(seed)


class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.value_head = nn.Sequential(
            nn.Linear(8, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1),
        )

    def forward(self, x, with_value=False):
        hid = torch.tanh(self.fc1(x))
        hid = torch.tanh(self.fc2(hid))
        hid = torch.tanh(self.fc3(hid))
        probabilities = F.softmax(self.fc4(hid), dim=-1)
        if with_value:
            return probabilities, self.value_head(x).squeeze(-1)
        return probabilities


class PolicyGradientAgent:
    def __init__(self, network, device=DEVICE):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        policy_parameters = [p for name, p in self.network.named_parameters()
                             if not name.startswith("value_head.")]
        self.optimizer = torch.optim.Adam([
            {"params": policy_parameters, "lr": LEARNING_RATE},
            {"params": self.network.value_head.parameters(), "lr": VALUE_LEARNING_RATE},
        ])

    @torch.no_grad()
    def sample(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        probabilities = self.network(state)
        if deterministic:
            return probabilities.argmax().item()
        return torch.multinomial(probabilities, num_samples=1).item()

    def learn(self, states, actions, returns):
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        probabilities, values = self.network(states, with_value=True)
        distribution = Categorical(probabilities)
        # 仍使用整局 Monte Carlo 回报；预测分支只作 baseline，不做自举或 PPO 更新。
        targets = returns / 100.0
        advantages = targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        policy_loss = -(distribution.log_prob(actions) * advantages).mean()
        value_loss = F.mse_loss(values, targets)
        loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * distribution.entropy().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        # 回报回归可重复拟合；策略只更新一次，避免用旧轨迹反复更新策略。
        for _ in range(VALUE_UPDATES - 1):
            value_loss = F.mse_loss(self.network.value_head(states).squeeze(-1), targets)
            self.optimizer.zero_grad(set_to_none=True)
            (0.5 * value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.value_head.parameters(), max_norm=1.0)
            self.optimizer.step()
        self.last_value_loss = value_loss.item()
        return loss.item()

    def save(self, path):
        torch.save({"network": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint["network"]
        shared_baseline = "value_head.weight" in state
        if shared_baseline:
            # 分离原先共享的特征，迁移后动作概率与回报预测保持一致。
            self.network.load_state_dict({k: v for k, v in state.items()
                                          if not k.startswith("value_head.")}, strict=False)
            for index, name in ((0, "fc1"), (2, "fc2"), (4, "fc3"), (6, "value_head")):
                self.network.value_head[index].load_state_dict(
                    {"weight": state[name + ".weight"], "bias": state[name + ".bias"]})
        else:
            self.network.load_state_dict(state)
        optimizer_state = checkpoint.get("optimizer")
        if not shared_baseline and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)


def evaluate(agent, episodes=100, seed_start=10000, deterministic=True):
    """使用独立种子和原版 1000 步上限评估，不影响训练随机数序列。"""
    scores, lengths = [], []
    successes, truncations = 0, 0
    devices = [next(agent.network.parameters()).device.index] if agent.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), gym.make("LunarLander-v3") as env:
        torch.manual_seed(10053)
        agent.network.eval()
        for episode in range(episodes):
            state, _ = env.reset(seed=seed_start + episode)
            total_reward = 0.0
            for step in range(1, env.spec.max_episode_steps + 1):
                state, reward, terminated, truncated, _ = env.step(agent.sample(state, deterministic))
                total_reward += reward
                if terminated or truncated:
                    successes += int(terminated and reward == 100)
                    truncations += int(truncated)
                    break
            scores.append(total_reward)
            lengths.append(step)
    agent.network.train()
    return {"mean_reward": float(np.mean(scores)), "std_reward": float(np.std(scores)),
            "mean_steps": float(np.mean(lengths)), "successes": successes,
            "truncations": truncations, "episodes": episodes, "rewards": scores,
            "deterministic": deterministic, "seed_start": seed_start}


def Train(num_batches=NUM_BATCH, output_dir=None, device=DEVICE,
          max_episode_steps=MAX_EPISODE_STEPS, eval_episodes=100, checkpoint=None):
    if num_batches < 1 or max_episode_steps < 1 or eval_episodes < 1:
        raise ValueError("batch 数、最大步数和评估轮数必须大于 0")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请将 DEVICE 设置为 cpu")
    if output_dir is None:
        output_dir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(output_dir)
    frame_dir = output_dir / "initial_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows, evaluations, initial_states = [], [], []
    best_score = (-1, -float("inf"))
    best_batch = 0

    with gym.make("LunarLander-v3", render_mode="rgb_array",
                  max_episode_steps=max_episode_steps) as env:
        fix(env, SEED)
        agent = PolicyGradientAgent(PolicyGradientNetwork(), device=device)
        if checkpoint is not None:
            agent.load(checkpoint)
        initial_evaluation = evaluate(agent, eval_episodes)
        best_score = (initial_evaluation["successes"], initial_evaluation["mean_reward"])
        agent.save(output_dir / "best_policy_gradient_agent.pth")
        print(f"初始评估: mean={initial_evaluation['mean_reward']:.1f}, "
              f"成功={initial_evaluation['successes']}/{eval_episodes}")
        print(f"Device: {agent.device}; 每轮最多 {max_episode_steps} 步; 输出: {output_dir}")
        start = time.perf_counter()
        progress = tqdm(range(num_batches), dynamic_ncols=True)
        for batch in progress:
            states, actions, returns = [], [], []
            total_rewards, final_rewards, lengths = [], [], []
            for episode in range(EPISODE_PER_BATCH):
                state, _ = env.reset()
                initial_states.append({"batch": batch + 1, "episode": episode + 1,
                                       **dict(zip(("x", "y", "vx", "vy", "angle", "angular_velocity",
                                                   "left_contact", "right_contact"), state.tolist()))})
                # 先保存初始画面，再执行动作；每轮一个文件。
                Image.fromarray(env.render()).save(
                    frame_dir / f"batch_{batch + 1:04d}_episode_{episode + 1:02d}.png")
                seq_rewards = []
                while True:
                    action = agent.sample(state)
                    states.append(state)
                    actions.append(action)
                    state, reward, terminated, truncated, _ = env.step(action)
                    seq_rewards.append(reward)
                    if terminated or truncated:
                        total_rewards.append(sum(seq_rewards))
                        final_rewards.append(reward)
                        lengths.append(len(seq_rewards))
                        break

                discounted_return = 0.0
                seq_returns = []
                for reward in reversed(seq_rewards):
                    discounted_return = reward + GAMMA * discounted_return
                    seq_returns.append(discounted_return)
                returns.extend(reversed(seq_returns))

            returns = np.asarray(returns, dtype=np.float32)
            loss = agent.learn(states, actions, returns)
            row = {"batch": batch + 1, "mean_reward": float(np.mean(total_rewards)),
                   "mean_final_reward": float(np.mean(final_rewards)),
                   "mean_steps": float(np.mean(lengths)), "total_steps": len(actions),
                   "loss": loss, "value_loss": agent.last_value_loss,
                   "elapsed_seconds": time.perf_counter() - start}
            rows.append(row)
            progress.set_description(f"Total: {row['mean_reward']:.1f}, Final: {row['mean_final_reward']:.1f}",
                                     refresh=False)
            progress.set_postfix(steps=len(actions), avg_steps=f"{row['mean_steps']:.0f}", device=str(agent.device))

            if (batch + 1) % EVAL_INTERVAL == 0 or batch + 1 == num_batches:
                result = evaluate(agent, eval_episodes)
                evaluations.append({"batch": batch + 1, **result})
                score = (result["successes"], result["mean_reward"])
                if score > best_score:
                    best_score, best_batch = score, batch + 1
                    agent.save(output_dir / "best_policy_gradient_agent.pth")
                agent.save(output_dir / "last_policy_gradient_agent.pth")
                (output_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2))
                with (output_dir / "training.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                with (output_dir / "initial_states.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=initial_states[0].keys())
                    writer.writeheader()
                    writer.writerows(initial_states)
                tqdm.write(f"阶段评估 batch={batch + 1}: mean={result['mean_reward']:.1f}, "
                           f"成功={result['successes']}/{eval_episodes}, 超时={result['truncations']}")
                if result["mean_reward"] >= 200 and result["successes"] >= 0.9 * eval_episodes:
                    best_batch = batch + 1
                    agent.save(output_dir / "best_policy_gradient_agent.pth")
                    tqdm.write("已达到验证目标，开始独立测试。")
                    break

        progress.close()
        elapsed = time.perf_counter() - start
        agent.load(output_dir / "best_policy_gradient_agent.pth")
        agent.save(output_dir / "policy_gradient_agent.pth")
        final_evaluation = evaluate(agent, 100, seed_start=30000)
        stochastic_evaluation = evaluate(agent, 100, seed_start=30000, deterministic=False)

    with (output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "initial_states.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=initial_states[0].keys())
        writer.writeheader()
        writer.writerows(initial_states)
    for key, filename, title in (
        ("mean_reward", "avg_total_rewards.png", "Average Total Reward per Batch"),
        ("mean_final_reward", "avg_final_rewards.png", "Average Final Reward per Batch"),
    ):
        plt.figure()
        plt.plot([row[key] for row in rows])
        plt.xlabel("Batch")
        plt.ylabel("Reward")
        plt.title(title)
        plt.savefig(output_dir / filename)
        plt.close()
    summary = {"seed": SEED, "batches": len(rows), "best_batch": best_batch,
               "network": [8, 128, 128, 64, 4], "entropy_coef": ENTROPY_COEF,
               "value_network": [8, 128, 128, 64, 1], "baseline_shared": False,
               "episodes_per_batch": EPISODE_PER_BATCH,
               "max_episode_steps": max_episode_steps, "gamma": GAMMA, "device": str(agent.device),
               "threads": torch.get_num_threads(), "optimizer": "Adam", "learning_rate": LEARNING_RATE,
               "value_learning_rate": VALUE_LEARNING_RATE, "initial_checkpoint": str(checkpoint),
               "value_updates_per_batch": VALUE_UPDATES,
               "elapsed_seconds": elapsed,
               "total_steps": sum(row["total_steps"] for row in rows),
               "initial_images": len(initial_states),
               "first_20_batch_reward": float(np.mean([r["mean_reward"] for r in rows[:20]])),
               "last_20_batch_reward": float(np.mean([r["mean_reward"] for r in rows[-20:]])),
               "initial_evaluation": initial_evaluation, "final_evaluation": final_evaluation,
               "stochastic_evaluation": stochastic_evaluation}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"训练用时 {elapsed:.1f}s，评估平均奖励 {final_evaluation['mean_reward']:.1f}，"
          f"成功着陆 {final_evaluation['successes']}/{final_evaluation['episodes']}")
    return summary


def custom_test(agent, action_list=None, episodes=5, actions_path=None):
    """固定种子测试，可保存动作以供 replay_test.py 独立回放。"""
    recordings = []
    with gym.make("LunarLander-v3") as env:
        seed = 543  # 到時候測試的 seed，請不要更改
        fix(env, seed)
        agent.network.eval()
        test_total_reward = []
        for episode in range(len(action_list) if action_list is not None else episodes):
            state, _ = env.reset()
            initial_state = state.tolist()
            total_reward = 0.0
            success = False
            terminated = truncated = False
            recorded_actions = []
            actions = action_list[episode] if action_list is not None else None
            for step in range(len(actions) if actions is not None else env.spec.max_episode_steps):
                action = int(actions[step] if actions is not None else agent.sample(state))
                recorded_actions.append(action)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    success = bool(terminated and reward == 100)
                    break
            print(f"Your reward is : {total_reward:.2f}, 成功降落：{'是' if success else '否'}")
            test_total_reward.append(total_reward)
            recordings.append({"initial_state": initial_state, "actions": recorded_actions,
                               "total_reward": total_reward, "success": success,
                               "terminated": bool(terminated), "truncated": bool(truncated)})
        if actions_path is not None:
            actions_path = Path(actions_path)
            actions_path.parent.mkdir(parents=True, exist_ok=True)
            actions_path.write_text(json.dumps({"seed": seed, "env_id": env.spec.id,
                "gymnasium_version": gym.__version__, "max_episode_steps": env.spec.max_episode_steps,
                "episodes": recordings}, indent=2), encoding="utf-8")
            print(f"动作已保存：{actions_path}")
    return test_total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE LunarLander 训练与评估")
    parser.add_argument("--checkpoint", type=Path, help="加载已有模型继续训练或评估")
    parser.add_argument("--eval-only", action="store_true", help="只评估模型，不训练")
    parser.add_argument("--custom-test", action="store_true", help="固定 seed=543 测试，打印每局奖励和是否成功降落")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--actions-out", type=Path,
                        default=Path(__file__).parent / "runs" / "custom_test_seed543.json",
                        help="自定义测试动作的保存位置")
    args = parser.parse_args()
    if args.custom_test:
        if args.checkpoint is None:
            parser.error("--custom-test 需要同时提供 --checkpoint")
        agent = PolicyGradientAgent(PolicyGradientNetwork(), device=args.device)
        agent.load(args.checkpoint)
        custom_test(agent, actions_path=args.actions_out)
    elif args.eval_only:
        if args.checkpoint is None:
            parser.error("--eval-only 需要同时提供 --checkpoint")
        torch.set_num_threads(1)
        agent = PolicyGradientAgent(PolicyGradientNetwork(), device=args.device)
        agent.load(args.checkpoint)
        for deterministic in (True, False):
            result = evaluate(agent, 100, seed_start=30000, deterministic=deterministic)
            print(json.dumps({k: v for k, v in result.items() if k != "rewards"}, indent=2))
    else:
        Train(checkpoint=args.checkpoint, device=args.device)
