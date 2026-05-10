import numpy as np
import random
import pickle


class QAgent:
    def __init__(self):

        # mastery x engagement x difficulty x actions
        self.q = np.zeros((3, 3, 3, 6))

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def choose_action(self, state):

        # explore
        if random.random() < self.epsilon:
            return random.randint(0, 5)

        # exploit
        return np.argmax(self.q[state])

    def update(self, state, action, reward, next_state):

        best_next = np.max(self.q[next_state])

        self.q[state][action] += self.alpha * (
            reward
            + self.gamma * best_next
            - self.q[state][action]
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)