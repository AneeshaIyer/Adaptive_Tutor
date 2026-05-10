import random

class Student:
    def __init__(self):
        # 0 = low, 1 = medium, 2 = high
        self.mastery = 1

        # 0 = disengaged, 1 = neutral, 2 = engaged
        self.engagement = 2

    def respond(self, difficulty):
        """
        difficulty:
        0 = easy
        1 = medium
        2 = hard
        """

        success_prob = {
            0: 0.9,
            1: 0.6,
            2: 0.3
        }[difficulty]

        correct = random.random() < success_prob

        if correct:
            self.mastery = min(2, self.mastery + 1)
            self.engagement = min(2, self.engagement + 1)
        else:
            self.engagement = max(0, self.engagement - 1)

        return correct