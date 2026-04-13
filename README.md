# 🚪 Automatic Door RL — Policy Iteration

> **Assignment 1** | Reinforcement Learning | Automatic Door Controller using Policy Iteration

---

## 📌 Problem Statement

Design a Reinforcement Learning model for an **automatic door** that decides whether to:
- **OPEN** the door
- **CLOSE** the door
- **WAIT** (do nothing)

The agent learns the optimal policy using **Policy Iteration** on a Markov Decision Process (MDP).

---

## 🧠 MDP Formulation

### States (12 total)
The state is a composite of three factors:

| Factor | Values |
|---|---|
| Door Position | `CLOSED`, `OPEN` |
| Person Presence | `NO_PERSON`, `DETECTED`, `PASSING` |
| Time of Day | `OFF_PEAK`, `PEAK` |

**Total states = 2 × 3 × 2 = 12**

### Actions (3)
| Action | Description |
|---|---|
| `OPEN` | Open the door |
| `CLOSE` | Close the door |
| `WAIT` | Hold current state |

### Reward Function

| Situation | Action | Reward |
|---|---|---|
| Person detected or passing | OPEN | **+10** |
| No person present | CLOSE | **+5** |
| Person passing, door still closed | WAIT | **-5** |
| No person, door opens | OPEN | **-3** |
| Closing on a passing person | CLOSE | **-8** |
| Person detected, cautious wait | WAIT | **+2** |
| Any state | WAIT (default) | **-1** |

### Transition Model
Stochastic transitions (probabilities < 1) model real-world uncertainty:
- Sensor noise (false negatives)
- Door mechanical lag
- Unpredictable pedestrian flow

### Discount Factor
`γ = 0.9` — values future rewards significantly while prioritising immediate outcomes.

---

## ⚙️ Algorithm: Policy Iteration

```
Initialize: π₀ (all WAIT), V₀ = 0

Repeat until stable:
  1. Policy Evaluation:
     V(s) ← Σ P(s'|s,π(s)) [R(s,π(s)) + γ·V(s')]
     (iterate until Δ < θ = 1e-8)

  2. Policy Improvement:
     π'(s) ← argmax_a Σ P(s'|s,a) [R(s,a) + γ·V(s')]

Until π' = π  →  Optimal Policy Found ✓
```

---

## 📁 Project Structure

```
automatic-door-rl/
├── src/
│   ├── door_environment.py   # MDP: states, actions, rewards, transitions
│   ├── policy_iteration.py   # Policy Iteration solver
│   └── visualize.py          # Result plots
├── tests/
│   └── test_door_rl.py       # Unit tests (pytest)
├── results/                  # Auto-generated plots
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/automatic-door-rl.git
cd automatic-door-rl
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Policy Iteration
```bash
cd src
python policy_iteration.py
```

### 4. Generate visualisations
```bash
python visualize.py
```

### 5. Run tests
```bash
cd ..
python -m pytest tests/ -v
```

---

## 📊 Results

After running, four plots are saved to `results/`:

| File | Description |
|---|---|
| `convergence.png` | Policy changes, ‖V‖ norm, and evaluation sweeps per iteration |
| `optimal_policy.png` | Optimal action and value function per state |
| `q_values.png` | Q(s,a) for all state–action pairs |
| `value_evolution.png` | How V(s) evolves over iterations |

### Key Findings (expected)

- **Closed + Person Passing** → always **OPEN** (high penalty for blocking)
- **Open + No Person** → always **CLOSE** (energy efficiency)
- **Open + Person Detected/Passing** → **OPEN** (keep open)
- **Closed + No Person** → **CLOSE** or **WAIT** (idle state)

---

## 📚 Concepts Demonstrated

- **Markov Decision Process** (MDP) formulation from a real-world problem
- **Policy Evaluation** via Bellman expectation equations
- **Policy Improvement** via greedy action selection
- **Policy Iteration** convergence guarantee
- **Stochastic transitions** and uncertainty modelling
- **Reward shaping** for multi-objective behaviour

---

## 🛠 Dependencies

```
numpy
matplotlib
pytest
```

---

## 📝 References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Puterman, M. L. (1994). *Markov Decision Processes*. Wiley.

---

*Assignment 1 — Reinforcement Learning | Policy Iteration*
