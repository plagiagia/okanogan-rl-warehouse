"""Milestones 3-4: train NFQ (M3) and DQN with replay + target network (M4) on CarRacing, same budget."""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import carracing as C

STEPS = int(os.environ.get("CAR_STEPS", 50_000))


def job(a):
    variant, seed = a
    torch.set_num_threads(1)
    t0 = time.time()
    q, hist = C.train(variant, total_steps=STEPS, seed=seed, log=lambda m: print(m, flush=True))
    ev = C.evaluate(q, episodes=10)
    rnd = C.evaluate(None, episodes=10, policy=lambda s: np.random.randint(5)) if seed == 0 and variant == "nfq" else None
    torch.save(q.state_dict(), f"models/car_{variant}_s{seed}.pt")
    res = dict(variant=variant, seed=seed, steps=STEPS, hist=hist, eval=ev, random_eval=rnd, minutes=(time.time() - t0) / 60)
    json.dump(res, open(f"results/car_{variant}_s{seed}.json", "w"))
    print("DONE", variant, seed, np.mean(ev), flush=True)
    return res


if __name__ == "__main__":
    jobs = [("nfq", 0), ("dqn", 0), ("nfq", 1), ("dqn", 1)]
    with Pool(2) as p:
        p.map(job, jobs, chunksize=1)
