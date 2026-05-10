import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.environment import AdaptiveTutorEnv
from agent.qlearning import QAgent

os.makedirs("graphs", exist_ok=True)

N_EPISODES = 100
STEPS = 30

class RandomAgent:
    def choose_action(self, state):
        return np.random.randint(6)

def run_policy(agent, label):
    env = AdaptiveTutorEnv()
    rewards = []
    for _ in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        for _ in range(STEPS):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    print(f"  {label:22s} | Avg Reward: {np.mean(rewards):+.2f} | Std: {np.std(rewards):.2f}")
    return rewards

# Load trained policies
random_agent = RandomAgent()

v1 = QAgent()
v1.load("models/qlearning_v1.pkl")
v1.epsilon = 0.0

v2 = QAgent()
v2.load("models/qlearning_v2.pkl")
v2.epsilon = 0.0

print("\n=== Evaluation Report ===")
r_random = run_policy(random_agent, "Random (Baseline)")
r_v1     = run_policy(v1,           "RL Policy v1")
r_v2     = run_policy(v2,           "RL Policy v2")

# Comparison table
print("\nPolicy Comparison Table:")
print(f"  {'Policy':<22} {'Avg Reward':>12} {'Improvement':>12}")
print("  " + "-" * 48)
base = np.mean(r_random)
for label, rewards in [("Random (Baseline)", r_random), ("RL v1", r_v1), ("RL v2", r_v2)]:
    avg = np.mean(rewards)
    imp = f"{(avg - base) / abs(base) * 100:+.1f}%" if label != "Random (Baseline)" else "—"
    print(f"  {label:<22} {avg:>12.2f} {imp:>12}")

# Find which log CSV exists
log_path = None
for candidate in ["logs/qlearning_v1.csv", "logs/qlearning_v2.csv", "logs/results.csv"]:
    if os.path.exists(candidate):
        log_path = candidate
        break

# Plot 1: Training reward curve
if log_path:
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], df["reward"], color="#7F77DD", linewidth=1.5, label="RL training reward")
    # smoothed line
    smoothed = df["reward"].rolling(20, min_periods=1).mean()
    plt.plot(df["episode"], smoothed, color="#3C3489", linewidth=2, label="Smoothed (20-ep avg)")
    plt.title("Adaptive Tutor — Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphs/reward_graph.png", dpi=150)
    plt.close()
    print("\nSaved → graphs/reward_graph.png")
else:
    print("\nNo training log found — skipping reward curve.")

# Plot 2: Baseline vs RL bar chart
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Random\n(Baseline)", "RL Policy v1", "RL Policy v2"]
means  = [np.mean(r_random), np.mean(r_v1), np.mean(r_v2)]
colors = ["#888780", "#7F77DD", "#1D9E75"]
bars   = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=11)
ax.set_title("Baseline vs RL Policy — Average Reward")
ax.set_ylabel("Avg Reward per Episode")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("graphs/evaluation_comparison.png", dpi=150)
plt.close()
print("Saved → graphs/evaluation_comparison.png")