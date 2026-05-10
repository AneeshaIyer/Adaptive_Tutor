from sim.student import Student

class AdaptiveTutorEnv:

    def __init__(self):
        self.student = Student()
        self.current_difficulty = 1

    def reset(self):
        self.student = Student()
        self.current_difficulty = 1
        return self.get_state()

    def get_state(self):
        return (
            self.student.mastery,
            self.student.engagement,
            self.current_difficulty
        )

    def step(self, action):
        """
        Actions:
        0 -> easy question
        1 -> medium question
        2 -> hard question
        3 -> give hint
        4 -> remediation lesson
        5 -> advance topic
        """

        if action in [0, 1, 2]:
            self.current_difficulty = action
            correct = self.student.respond(action)

            reward = 10 if correct else -5

            if self.student.mastery == 2:
                reward += 15

            if self.student.engagement == 0:
                reward -= 20

        elif action == 3:
            reward = 5

        elif action == 4:
            reward = 8

        elif action == 5:
            reward = 12

        next_state = self.get_state()
        done = False

        return next_state, reward, done