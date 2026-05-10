from sim.environment import AdaptiveTutorEnv
from agent.qlearning import QAgent
import pandas as pd
import uuid
import os
import yaml


os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

#CONFIG_PATH = "configs/qlearning_v1.yaml"
CONFIG_PATH = "configs/qlearning_v2.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


env = AdaptiveTutorEnv()
agent = QAgent()

agent.alpha = config["alpha"]
agent.gamma = config["gamma"]
agent.epsilon = config["epsilon"]
agent.epsilon_decay = config["epsilon_decay"]
agent.epsilon_min = config["epsilon_min"]

episodes = config["episodes"]
steps_per_episode = config["steps_per_episode"]

run_id = str(uuid.uuid4())
logs = []


for ep in range(episodes):

    state = env.reset()
    total_reward = 0

    for _ in range(steps_per_episode):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state
        total_reward += reward

    agent.epsilon = max(
        agent.epsilon_min,
        agent.epsilon * agent.epsilon_decay
    )

    logs.append({
        "run_id": run_id,
        "episode": ep,
        "reward": total_reward,
        "epsilon": agent.epsilon,
        "alpha": agent.alpha,
        "gamma": agent.gamma
    })

    if ep % 50 == 0:
        print(
            f"Episode {ep} | "
            f"Reward={total_reward} | "
            f"Epsilon={agent.epsilon:.3f}"
        )


model_name = os.path.basename(CONFIG_PATH).replace(".yaml", ".pkl")
agent.save(f"models/{model_name}")

csv_name = os.path.basename(CONFIG_PATH).replace(".yaml", ".csv")

pd.DataFrame(logs).to_csv(
    f"logs/{csv_name}",
    index=False
)
print("\nTraining complete!")