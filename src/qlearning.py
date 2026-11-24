"""Tabular Q-learning with epsilon-greedy exploration."""
import numpy as np


def train_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
                     eps_start=1.0, eps_end=0.05, eps_decay=0.999, seed=0):
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    eps = eps_start
    returns, lengths, success = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            a = env.action_space.sample() if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = env.step(a)
            target = r + (0.0 if term else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s, total, steps, done = s2, total + r, steps + 1, term or trunc
        eps = max(eps_end, eps * eps_decay)
        returns.append(total); lengths.append(steps); success.append(bool(term))
    return Q, np.array(returns), np.array(lengths), np.array(success)


def evaluate_q(env, Q, episodes=200, seed=10_000):
    returns, lengths, success = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            total, steps, done = total + r, steps + 1, term or trunc
        returns.append(total); lengths.append(steps); success.append(bool(term))
    return dict(mean_return=float(np.mean(returns)), mean_steps=float(np.mean(lengths)),
                success_rate=float(np.mean(success)))


def evaluate_random(env, episodes=200, seed=10_000):
    env.action_space.seed(seed)
    returns, lengths, success = [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total, steps, done = total + r, steps + 1, term or trunc
        returns.append(total); lengths.append(steps); success.append(bool(term))
    return dict(mean_return=float(np.mean(returns)), mean_steps=float(np.mean(lengths)),
                success_rate=float(np.mean(success)))
