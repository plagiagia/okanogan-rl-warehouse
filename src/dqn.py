"""Minimal DQN with switchable experience replay and target network (for ablation)."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def train_dqn(env, obs_fn, obs_dim, total_steps=60_000, use_replay=True, use_target=True,
              gamma=0.99, lr=1e-3, batch=64, buffer=20_000, target_sync=500,
              eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5, seed=0, warmup=500):
    """obs_fn maps a raw env observation to a float32 vector."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n_actions = env.action_space.n
    q = QNet(obs_dim, n_actions)
    tgt = QNet(obs_dim, n_actions); tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    mem = deque(maxlen=buffer)
    steps_total, ep = 0, 0
    returns, delivered, ends = [], [], []
    while steps_total < total_steps:
        o, _ = env.reset(seed=seed * 1_000_000 + ep); ep += 1
        x = obs_fn(o)
        done, total = False, 0.0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * steps_total / (eps_decay_frac * total_steps))
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(q(torch.as_tensor(x)).argmax())
            o2, r, term, trunc, info = env.step(a)
            x2 = obs_fn(o2)
            transition = (x, a, r, x2, float(term))
            if use_replay:
                mem.append(transition)
                ready = len(mem) >= max(batch, warmup)
                b = random.sample(mem, batch) if ready else None
            else:
                b = [transition]          # naive: learn from the single latest step
            if b is not None:
                xs, as_, rs, x2s, ds = zip(*b)
                xs = torch.as_tensor(np.stack(xs)); x2s = torch.as_tensor(np.stack(x2s))
                as_ = torch.as_tensor(as_).unsqueeze(1)
                rs = torch.as_tensor(rs, dtype=torch.float32); ds = torch.as_tensor(ds, dtype=torch.float32)
                with torch.no_grad():
                    nxt = (tgt if use_target else q)(x2s).max(1).values
                    y = rs + gamma * (1 - ds) * nxt
                pred = q(xs).gather(1, as_).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
            steps_total += 1
            if use_target and steps_total % target_sync == 0:
                tgt.load_state_dict(q.state_dict())
            x, total, done = x2, total + r, term or trunc
        returns.append(total); delivered.append(bool(info.get("delivered", False))); ends.append(steps_total)
    return q, np.array(returns), np.array(delivered), np.array(ends)


def greedy_action(q, x):
    with torch.no_grad():
        return int(q(torch.as_tensor(x)).argmax())
