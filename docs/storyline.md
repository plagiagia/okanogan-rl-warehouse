# Presentation storyline (5-7 min)

1. **Hook (30s):** "A warehouse cart that must deliver and never hit a shelf. Can it learn that?"
2. **Milestone 1-2 (60s):** a random driver completes ~2-3% of trips; a Q-table solves Taxi (100%, 12.8 steps). Chart: `m2_q_learning_curve.png`.
3. **Milestone 3-4 (90s):** from pixels in CarRacing. Play `m4_dqn_episode.gif`. Chart: `m4_comparison.png`. Message: replay + target gave faster, steadier learning, not a higher final score.
4. **The twist (60s):** the NFQ seed that trained to ~490 scored -19.9 greedily (steers left forever). "Training curves lie."
5. **Milestone 5 (90s):** our own warehouse. Naive DQN 0.3%, replay 45.6%. Play `m5c_trained_robot.gif`.
6. **Verdict (60s):** 36% delivery, all failures are loops; safety mask gives 0 bumps by construction. For a static warehouse use a planner.

Rule: one chart per act, one number per chart. Lead with the failure analysis, the part nobody else shows.
