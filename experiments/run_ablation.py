"""DQN ablation on the warehouse env: naive vs +replay vs +target vs both, several seeds."""
import json, sys, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.dqn import train_dqn, greedy_action
from src.warehouse_env import WarehouseEnv, evaluate_policy

STEPS = 120_000
VARIANTS = {"naive": (False, False), "replay_only": (True, False),
            "target_only": (False, True), "replay+target": (True, True)}
SEEDS = [0, 1, 2]
GRID = list(range(5_000, STEPS + 1, 5_000))


def job(args):
    name, seed = args
    torch.set_num_threads(1)
    rp, tg = VARIANTS[name]
    env = WarehouseEnv()
    q, R, D, E = train_dqn(env, lambda o: o, 11, total_steps=STEPS, use_replay=rp, use_target=tg, seed=seed)
    curve = []
    for g in GRID:                       # rolling delivery rate over the last 100 episodes before step g
        idx = np.where(E <= g)[0][-100:]
        curve.append(float(D[idx].mean()) if len(idx) else 0.0)
    ev = evaluate_policy(env, lambda o: greedy_action(q, o), episodes=300)
    if name == "replay+target" and seed == 0:
        torch.save(q.state_dict(), "models/dqn_replay_target_seed0.pt")
    print(name, seed, ev, flush=True)
    return dict(variant=name, seed=seed, curve=curve, eval=ev, episodes=int(len(R)))


if __name__ == "__main__":
    jobs = [(n, s) for s in SEEDS for n in VARIANTS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(dict(steps=STEPS, grid=GRID, runs=res), open("results/ablation.json", "w"))
    print("done")
