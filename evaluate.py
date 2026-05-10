import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/results.csv")

print(df.head())

plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["reward"])

plt.title("Adaptive Tutor Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.grid(True)
plt.show()