"""CarRacing pipeline for milestones 3-4: pixel preprocessing, CNN Q-network, two training variants.

variant "nfq": online neural fitted Q-learning (one sample per update, bootstraps from the same network)
variant "dqn": + experience replay + target network (milestone 4 improvements)
"""
import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

N_ACTIONS = 5          # CarRacing discrete: 0 noop, 1 steer right, 2 steer left, 3 gas, 4 brake
SKIP, STACK, PATIENCE, MAX_STEPS = 4, 4, 50, 250   # action repeat, frames stacked, no-progress limit, episode cap (agent steps)


def prep(frame):
    """96x96x3 RGB -> 42x48 grayscale uint8 (drop the bottom status bar, downsample x2)."""
    g = (frame[:84].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    g = g.reshape(42, 2, 48, 2).mean((1, 3)) if g.shape[1] == 96 else g
    return g.astype(np.uint8)


class CarEnv:
    """Wraps CarRacing: skips the zoom-in intro, repeats each action 4x, stacks 4 frames,
    scales reward by 0.1 (clipped at -5) and ends the episode after 50 agent steps without a tile reward."""

    def __init__(self, seed=0, render=False):
        self.env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array" if render else None)
        self.seed = seed
        self.frames = deque(maxlen=STACK)

    def reset(self, seed=None):
        o, _ = self.env.reset(seed=seed)
        for _ in range(50):
            o, *_ = self.env.step(0)
        f = prep(o)
        for _ in range(STACK):
            self.frames.append(f)
        self.t, self.idle, self.raw_return = 0, 0, 0.0
        return np.stack(self.frames)

    def step(self, a):
        total, done = 0.0, False
        for _ in range(SKIP):
            o, r, te, tr, _ = self.env.step(int(a))
            total += r
            done = done or te or tr
            if done:
                break
        self.frames.append(prep(o))
        self.t += 1
        self.raw_return += total
        self.idle = 0 if total > 0 else self.idle + 1
        done = done or self.idle >= PATIENCE or self.t >= MAX_STEPS
        return np.stack(self.frames), float(np.clip(total * 0.1, -5, 5)), done

    def render_rgb(self):
        return self.env.render() if self.env.render_mode else None


class QCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(STACK, 16, 4, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, STACK, 42, 48)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS))

    def forward(self, x):                       # x: uint8 (B, 4, 42, 48)
        return self.head(self.conv(x.float() / 255.0))


def act_greedy(q, s):
    with torch.no_grad():
        return int(q(torch.as_tensor(s).unsqueeze(0)).argmax())


def train(variant="dqn", total_steps=50_000, seed=0, gamma=0.97, lr=1e-4, batch=32, buffer=20_000,
          target_sync=1000, update_every=4, warmup=1000, eps_end=0.05, eps_frac=0.6, log=print):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = CarEnv()
    q = QCNN(); opt = torch.optim.Adam(q.parameters(), lr=lr)
    use_dqn = variant == "dqn"
    tgt = QCNN(); tgt.load_state_dict(q.state_dict())
    mem = deque(maxlen=buffer)
    step, ep, hist = 0, 0, []
    while step < total_steps:
        s = env.reset(seed=seed * 10_000 + ep); ep += 1; done = False
        while not done:
            eps = max(eps_end, 1.0 - (1.0 - eps_end) * step / (eps_frac * total_steps))
            a = random.randrange(N_ACTIONS) if random.random() < eps else act_greedy(q, s)
            s2, r, done = env.step(a); step += 1
            if use_dqn:
                mem.append((s, a, r, s2, float(done)))
                if len(mem) >= warmup and step % update_every == 0:
                    b = random.sample(mem, batch)
                else:
                    b = None
            else:
                b = [(s, a, r, s2, float(done))]          # online: learn from the single latest transition
            if b is not None:
                S, A, R, S2, D = zip(*b)
                S = torch.as_tensor(np.stack(S)); S2 = torch.as_tensor(np.stack(S2))
                A = torch.as_tensor(A).unsqueeze(1); R = torch.as_tensor(R, dtype=torch.float32)
                D = torch.as_tensor(D, dtype=torch.float32)
                with torch.no_grad():
                    y = R + gamma * (1 - D) * (tgt if use_dqn else q)(S2).max(1).values
                loss = nn.functional.smooth_l1_loss(q(S).gather(1, A).squeeze(1), y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            if use_dqn and step % target_sync == 0:
                tgt.load_state_dict(q.state_dict())
            s = s2
        hist.append((step, env.raw_return))
        if ep % 20 == 0:
            log(f"{variant} s{seed} ep {ep} step {step} eps {eps:.2f} avg raw return (last 20) "
                f"{np.mean([h[1] for h in hist[-20:]]):.1f}")
    return q, hist


def evaluate(q, episodes=10, seed=900_000, policy=None):
    """Raw CarRacing return per episode with the greedy policy (or a custom policy(s)->a)."""
    env = CarEnv(); out = []
    for i in range(episodes):
        s = env.reset(seed=seed + i); done = False
        while not done:
            a = policy(s) if policy else act_greedy(q, s)
            s, _, done = env.step(a)
        out.append(env.raw_return)
    return out
