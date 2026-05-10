# Adaptive Learning Tutor using Reinforcement Learning and MLOps

## Overview
This project builds an intelligent adaptive tutoring system using Reinforcement Learning (Q-learning). The tutor dynamically adjusts question difficulty, hints, and remediation content based on student performance and engagement.

The goal is to maximize learning outcomes while minimizing disengagement and frustration.

---

## SDG Mapping
**SDG 4 – Quality Education**

This project supports SDG 4 by personalizing educational experiences, improving concept mastery, and making learning more adaptive for students with different abilities.

---

## Problem Statement
Traditional tutoring systems follow fixed teaching sequences.

This project develops an RL-based adaptive tutor that learns:

> **What should be taught next?**

based on:

- student mastery
- student engagement
- learning difficulty

The objective is to optimize learning progression.

---

## Reinforcement Learning Methodology

### Algorithm Used
**Q-learning**

Reason:
- discrete state space
- simple implementation
- interpretable policy learning

### State Space
State consists of:

- mastery level (low / medium / high)
- engagement level (disengaged / neutral / engaged)
- current difficulty (easy / medium / hard)

Example:

```python
(1, 2, 0)
```

### Action Space
The tutor can choose:

0 → Easy Question  
1 → Medium Question  
2 → Hard Question  
3 → Give Hint  
4 → Remediation Lesson  
5 → Advance Topic

### Reward Function
Positive reward:
- correct answer
- mastery increase
- engagement increase

Negative reward:
- repeated mistakes
- disengagement

---

## Project Structure
```text
AdaptiveTutor/
│
├── sim/
├── agent/
├── configs/
├── experiments/
├── models/
├── logs/
├── train.py
├── evaluate.py
└── README.md
```

---

## Experiments
### Version 1
Config:
- alpha = 0.1
- gamma = 0.95
- epsilon decay = 0.995

### Version 2
Config:
- alpha = 0.2
- gamma = 0.99
- epsilon decay = 0.990

Version 2 showed improved reward and faster convergence.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Evaluate:

```bash
python evaluate.py
```

---

## Monitoring Plan
If deployed in production, monitor:

- learner engagement score
- concept mastery progression
- completion rate
- dropout probability
- average learning time

---

## Conclusion
Adaptive Reinforcement Learning improves personalization in tutoring systems and creates a scalable path toward intelligent educational platforms.