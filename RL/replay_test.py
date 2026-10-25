"""回放 train_improved.py --custom-test 保存的动作；不加载模型、不训练。"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame


def replay(actions_path, fps=50, verify_only=False, frame_skip=2):
    data = json.loads(Path(actions_path).read_text(encoding="utf-8"))
    paused = False
    try:
        if not verify_only:
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode((600, 496))
            pygame.display.set_caption("LunarLander replay | Space: pause | Esc: quit")
            font = pygame.font.Font(None, 24)
            clock = pygame.time.Clock()

        def handle_events():
            nonlocal paused
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            pygame.display.set_caption(
                                "LunarLander replay | " + ("PAUSED | " if paused else "")
                                + "Space: pause/resume | Esc: quit")
                if not paused:
                    return True
                clock.tick(30)

        def draw(frame, episode, step, action, reward, status):
            surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.fill((20, 23, 30))
            screen.blit(surface, (0, 96))
            engine = ("Off", "Left engine", "Main engine", "Right engine")[action]
            lines = (
                f"Episode {episode}/{len(data['episodes'])}    Step {step}    Action: {engine}",
                f"Reward: {reward:.2f}    Landed: {status}",
                "Space: pause/resume    Esc: quit",
            )
            for row, line in enumerate(lines):
                screen.blit(font.render(line, True, (235, 240, 245)), (16, 8 + row * 28))
            pygame.display.flip()

        with gym.make(data["env_id"], render_mode=None if verify_only else "rgb_array",
                      max_episode_steps=data["max_episode_steps"]) as env:
            # fix(env, 543) 曾消费一次 reset；这里保持完全相同的随机数顺序。
            env.reset(seed=data["seed"])
            for episode, record in enumerate(data["episodes"], 1):
                state, _ = env.reset()
                if not np.allclose(state, record["initial_state"], rtol=0, atol=1e-6):
                    raise RuntimeError(f"第 {episode} 局初始状态不一致，请使用采集时的环境版本。")
                total_reward = 0.0
                success = False
                terminated = truncated = False
                if not verify_only:
                    draw(env.render(), episode, 0, 0, total_reward, "playing")
                for step, action in enumerate(record["actions"], 1):
                    if not verify_only and not handle_events():
                        return
                    _, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    success = bool(terminated and reward == 100)
                    if not verify_only:
                        status = ("YES" if success else "NO") if terminated or truncated else "playing"
                        if step == len(record["actions"]) and not (terminated or truncated):
                            status = "NO (actions ended)"
                        if step % frame_skip == 0 or terminated or truncated or step == len(record["actions"]):
                            draw(env.render(), episode, step, action, total_reward, status)
                        clock.tick(fps)
                    if terminated or truncated:
                        break
                if (abs(total_reward - record["total_reward"]) > 1e-5
                        or success != record["success"]
                        or bool(terminated) != record["terminated"]
                        or bool(truncated) != record["truncated"]):
                    raise RuntimeError(f"第 {episode} 局回放结果与记录不一致。")
                print(f"第 {episode} 局 reward: {total_reward:.2f}, 成功降落：{'是' if success else '否'}",
                      flush=True)
                if not verify_only:
                    # 每局结束后保留结果画面 1.5 秒，最后一局结束后关闭窗口。
                    for _ in range(45):
                        if not handle_events():
                            return
                        clock.tick(30)
    finally:
        if not verify_only:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="弹窗动画回放固定 seed=543 的测试动作")
    parser.add_argument("actions", nargs="?", type=Path,
                        default=Path(__file__).parent / "runs" / "custom_test_seed543.json")
    parser.add_argument("--fps", type=int, default=50, help="每秒步数；降低数值可慢放，默认 50")
    parser.add_argument("--frame-skip", type=int, default=2, help="每 N 步刷新画面，所有操作仍完整执行；远程显示慢时可用 5")
    parser.add_argument("--verify-only", action="store_true", help="不打开窗口，只校验回放结果")
    args = parser.parse_args()
    if args.fps <= 0 or args.frame_skip <= 0:
        parser.error("--fps 和 --frame-skip 必须大于 0")
    replay(args.actions, fps=args.fps, verify_only=args.verify_only, frame_skip=args.frame_skip)
