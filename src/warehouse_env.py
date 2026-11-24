"""Custom Okanogan warehouse: a grid with shelves; pick up an item, deliver to a target, never hit a shelf."""
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

LAYOUT = [
    "..........",
    ".##..##.#.",
    ".##..##.#.",
    ".##..##.#.",
    "..........",
    ".##..##.#.",
    ".##..##.#.",
    ".##..##.#.",
    "..........",
    "..........",
]
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right (dx, dy)


class WarehouseEnv(gym.Env):
    """Observation (float32, 11 dims): robot xy, item xy, target xy, has_item, 4 blocked-neighbour flags
    (all positions normalised to [0,1]). Reward: -1/step, small progress shaping, +10 pickup,
    +50 delivery, -5 for bumping a shelf or wall (robot stays put, episode continues, collision counted)."""

    def __init__(self, max_steps=80, shaping=0.2):
        self.grid = np.array([[c == "#" for c in row] for row in LAYOUT])
        self.h, self.w = self.grid.shape
        self.free = [(x, y) for y in range(self.h) for x in range(self.w) if not self.grid[y, x]]
        self.max_steps, self.shaping = max_steps, shaping
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (11,), np.float32)

    def blocked(self, x, y):
        return not (0 <= x < self.w and 0 <= y < self.h) or self.grid[y, x]

    def _dist(self, a, b):  # BFS path length on the grid
        if a == b:
            return 0
        seen, q = {a}, deque([(a, 0)])
        while q:
            (x, y), d = q.popleft()
            for dx, dy in MOVES:
                n = (x + dx, y + dy)
                if n == b:
                    return d + 1
                if n not in seen and not self.blocked(*n):
                    seen.add(n); q.append((n, d + 1))
        return 10 ** 6

    def optimal_steps(self):
        return self._dist(self.robot, self.item) + self._dist(self.item, self.target)

    def _goal(self):
        return self.target if self.has_item else self.item

    def _obs(self):
        nb = [float(self.blocked(self.robot[0] + dx, self.robot[1] + dy)) for dx, dy in MOVES]
        sx, sy = self.w - 1, self.h - 1
        return np.array([self.robot[0] / sx, self.robot[1] / sy, self.item[0] / sx, self.item[1] / sy,
                         self.target[0] / sx, self.target[1] / sy, float(self.has_item), *nb], np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        i = r.choice(len(self.free), 3, replace=False)
        self.robot, self.item, self.target = (self.free[j] for j in i)
        self.has_item, self.t, self.collisions = False, 0, 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        dx, dy = MOVES[int(action)]
        nx, ny = self.robot[0] + dx, self.robot[1] + dy
        info = {"collision": False, "delivered": False}
        if self.blocked(nx, ny):
            self.collisions += 1
            info["collision"] = True
            return self._obs(), -5.0, False, self.t >= self.max_steps, info
        before = self._dist(self.robot, self._goal())
        self.robot = (nx, ny)
        reward = -1.0
        if not self.has_item and self.robot == self.item:
            self.has_item, reward = True, reward + 10.0
        elif self.has_item and self.robot == self.target:
            info["delivered"] = True
            return self._obs(), reward + 50.0, True, False, info
        reward += self.shaping * (before - self._dist(self.robot, self._goal()))
        return self._obs(), reward, False, self.t >= self.max_steps, info


def evaluate_policy(env, policy, episodes=300, seed=50_000):
    """policy(obs) -> action. collision_rate = share of episodes with at least one bump."""
    deliv = coll = 0
    eff, bumps = [], []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        opt = env.optimal_steps()
        done, steps = False, 0
        while not done:
            o, r, term, trunc, info = env.step(policy(o))
            steps += 1; done = term or trunc
        deliv += info["delivered"]; coll += env.collisions > 0; bumps.append(env.collisions)
        if info["delivered"]:
            eff.append(opt / steps)
    n = episodes
    return dict(delivery_rate=deliv / n, collision_rate=coll / n, bumps_per_episode=float(np.mean(bumps)),
                path_efficiency=float(np.mean(eff)) if eff else 0.0)
