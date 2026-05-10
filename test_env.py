from sim.environment import AdaptiveTutorEnv

env = AdaptiveTutorEnv()

state = env.reset()
print("Initial state:", state)

next_state, reward, done = env.step(0)

print("Next state:", next_state)
print("Reward:", reward)
print("Done:", done)