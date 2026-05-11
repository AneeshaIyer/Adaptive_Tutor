"""
demo.py  —  Visual walkthrough of the Adaptive Tutor RL policy
Shows exactly what action the agent picks for each student state,
and how the student responds.

Usage:
    python demo.py
"""

import os
import time
from sim.environment import AdaptiveTutorEnv
from agent.qlearning import QAgent

POLICY_PATH = "models/qlearning_v1.pkl"
EPISODES    = 3
STEPS       = 10

MASTERY_LABELS    = {0: "Low", 1: "Medium", 2: "High"}
ENGAGEMENT_LABELS = {0: "Disengaged", 1: "Neutral", 2: "Engaged"}
DIFFICULTY_LABELS = {0: "Easy", 1: "Medium", 2: "Hard"}
ACTION_LABELS     = {
    0: "Easy Question",
    1: "Medium Question",
    2: "Hard Question",
    3: "Give Hint",
    4: "Remediation Lesson",
    5: "Advance Topic",
}

def print_separator():
    print("─" * 60)

def run_demo():
    if not os.path.exists(POLICY_PATH):
        print(f"Policy not found at {POLICY_PATH}. Run train.py first.")
        return

    agent = QAgent()
    agent.load(POLICY_PATH)
    agent.epsilon = 0.0    # pure exploitation — no randomness

    env = AdaptiveTutorEnv()

    for ep in range(1, EPISODES + 1):
        state        = env.reset()
        total_reward = 0.0

        print(f"\n{'='*60}")
        print(f"  STUDENT SESSION {ep}")
        print(f"{'='*60}")
        print(f"  Starting state:")
        print(f"    Mastery    : {MASTERY_LABELS[state[0]]}")
        print(f"    Engagement : {ENGAGEMENT_LABELS[state[1]]}")
        print(f"    Difficulty : {DIFFICULTY_LABELS[state[2]]}")
        print_separator()

        for step in range(1, STEPS + 1):
            action     = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            result  = "✓ Correct" if reward > 0 else "✗ Wrong"
            mastery_change    = ""
            engagement_change = ""

            if next_state[0] > state[0]:
                mastery_change = " ↑"
            elif next_state[0] < state[0]:
                mastery_change = " ↓"

            if next_state[1] > state[1]:
                engagement_change = " ↑"
            elif next_state[1] < state[1]:
                engagement_change = " ↓"

            print(f"  Step {step:2d}")
            print(f"    State      : mastery={MASTERY_LABELS[state[0]]}, "
                  f"engagement={ENGAGEMENT_LABELS[state[1]]}, "
                  f"difficulty={DIFFICULTY_LABELS[state[2]]}")
            print(f"    Agent picks: {ACTION_LABELS[action]}")
            print(f"    Outcome    : {result}  |  reward={reward:+.1f}")
            print(f"    New state  : mastery={MASTERY_LABELS[next_state[0]]}{mastery_change}, "
                  f"engagement={ENGAGEMENT_LABELS[next_state[1]]}{engagement_change}")
            print_separator()

            state = next_state
            if done:
                break

        print(f"  Session {ep} complete — Total reward: {total_reward:+.1f}")

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_demo()