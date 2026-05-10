from agent.qlearning import QAgent

agent = QAgent()

state = (1, 2, 1)

action = agent.choose_action(state)

print("Chosen action:", action)

agent.update(
    state,
    action,
    reward=10,
    next_state=(2, 2, 1)
)

print("Updated Q value:")
print(agent.q[state][action])