# Presentation storyline (5-7 min)

1. **Hook (30s):** "A warehouse cart that must deliver and never hit a shelf. Can it learn that?"
2. **Framing (45s):** state, action, reward. Random driver completes ~3% of trips in the Taxi sandbox.
3. **Act 1, a table is enough (45s):** Q-learning reaches 100% in ~13 steps. Chart: `ch2_q_learning_curve.png`.
4. **Act 2, the table can't cover the real warehouse (60s):** 656,880 states vs a 120k-step budget; naive DQN delivers 0.3%. Mention the "learned to crash" reward-design lesson.
5. **Act 3, what stabilises DQN (90s):** `ch4_ablation.png`. Replay is the hero (45.6%); target network alone does nothing (1.6%).
6. **Act 4, the honest verdict (90s):** play the GIF, then the table: 36% delivery, all failures are loops. "Training curves said 90%; greedy evaluation said 36%."
7. **Close (30s):** safety mask gives 0 collisions by construction; for a static warehouse use a planner; RL needs a shield.

Rule: one chart per act, one number per chart. Lead with the failure analysis, since that is the part nobody else shows.
