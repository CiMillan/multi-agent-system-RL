# Coexistence of Imitation Learning and Reinforcement Learning in Cooperation Evolution

This repository contains a Python-based multi-agent simulation framework replicating and exploring the dynamics described in the paper:

> **Evolution of cooperation guided by the coexistence of imitation learning and reinforcement learning**  
> *Wei Tang, Guoling Wang, and Zhiyan Xing*  
> Published in *Scientific Reports* (2025), Nature Portfolio  
> [Search on Google Scholar](https://scholar.google.com/scholar?q=Evolution+of+cooperation+guided+by+the+coexistence+of+imitation+learning+and+reinforcement+learning&hl=en&as_sdt=0,5)

---

## 🔬 Scientific Overview

In evolutionary game theory, cooperation is traditionally modeled using **Imitation Learning (IL)**, where agents adopt strategies from successful neighbors. This research introduces a hybrid model combining IL with **Reinforcement Learning (RL)**, where agents also adapt autonomously based on personal experience and feedback.

The coexistence of both paradigms is studied across three classic games:
1. **Prisoner's Dilemma Game (PDG)**
2. **Coordination Game (CG)**
3. **Coexistence Game (CoG) / Hawk-Dove / Snowdrift**

Simulations are executed across two population structures:
* **Well-mixed populations** (fully connected networks)
* **Square lattices** (grid topologies where interactions are local)

---

## ✅ Validation Status

This codebase has been checked against the paper's reported baseline results (well-mixed and square-lattice, all three games). Three bugs were found and fixed; one discrepancy remains open.

**Fixed and verified:**
| Bug | Where | Fix |
|---|---|---|
| RL experience-weight formula inverted: code used `(1-ρ)·H+1` instead of the paper's `H(t)=ρ·H(t-1)+1` | `main.py`, `mlcoop/core/rl.py` | Commit `fd968a2` |
| `baselines.py` crashed (`RLParams` has no `alpha` field) and had φ/δ swapped relative to the paper | `mlcoop/experiments/baselines.py` | Commit `a757cdd` |

Post-fix results vs. the paper's reported figures:

| Game | Well-mixed (paper → code) | Square lattice (paper → code) |
|---|---|---|
| PDG | ~42% → 42.00% | ~44% → 43.50% |
| CG | ~35% → 32.56% | ~40% → **34.3-35.0%** ⚠️ |
| CoG | ~100% → 100.00% | → 100% by iter 200 → 100.00% |

**Open issue:** the square-lattice CG result does not currently reproduce the paper (~5-6pp gap, confirmed stable across multiple seeds — not random variance). Systematically tested and ruled out: which opposite-strategy neighbor is imitated (irrelevant with binary strategies), and the Bernoulli-vs-exact-count sampling rule (moves the result by <1pp). A stricter eligibility comparison (`<` instead of `≤`) closes about a third of the gap but isn't clearly what the paper specifies either way. Likely cause: an IL/RL timing-interaction detail under the lattice's local structure, or a node-degree boundary effect — not yet resolved. **Do not treat lattice-CG results from this codebase as paper-verified.** PDG and CoG lattice results are solid.

---

## 📁 Repository Structure

The project is structured to offer both a simple, self-contained single-file script and a modular, clean package:

* **[main.py](main.py)**: A research-clean, single-file implementation scaffold containing all models, payoff configurations, topologies, and simulation drivers in one place.
* **[ml_package/](ml_package)**: A modularized version of the simulation framework package (`mlcoop`):
  * `mlcoop/core/config.py`: Dataclasses for RL, IL, and ML parameters.
  * `mlcoop/core/games.py`: Game payoff matrices (PDG, CG, CoG).
  * `mlcoop/core/topology.py`: Graph creation (square lattices) and neighborhood mappings.
  * `mlcoop/core/payoffs.py`: Average and counterfactual payoff calculations.
  * `mlcoop/core/il.py` / `rl.py` / `ml.py`: Step update rules for imitation, reinforcement, and conflict resolution.
  * `mlcoop/core/simulators.py`: High-level simulation runners.
  * `mlcoop/experiments/baselines.py`: Scripts to run baseline sweeps.
* **[mainV1/](mainV1)**: Legacy/v1 version of the single-file simulator.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:CiMillan/multi-agent-system-RL.git
   cd multi-agent-system-RL
   ```

2. **Install dependencies**:
   This simulation relies on standard scientific Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires `numpy`, `networkx`, and optionally `matplotlib` for plotting).*

---

## 🚀 Running Simulations

### 1. Single-File Scaffold
To run the self-contained baseline simulations:
```bash
python3 main.py
```

### 2. Modular Package
To run or extend using the modular package structure:
```python
from ml_package.mlcoop.experiments.baselines import run_well_mixed_baseline

# Run a baseline simulation for the Prisoner's Dilemma
results = run_well_mixed_baseline("PDG", seed=42)
print("Final MFC (Mean Fraction of Cooperators):", results["mfc"][-1])
```

---

## ⚠️ Handling Paper Ambiguities

To make the codebase robust and easy to tune, we have isolated these assumptions behind explicit comments marked as:
`# AMBIGUITY FROM PAPER`

Key areas with assumptions, updated with what the validation pass above actually found:
1. **EWA attraction updates**: The update algebra in `rl_pending_strategy` uses a standard Experience-Weighted Attraction (EWA) model. The H(t) recursion within it was wrong until commit `fd968a2` — see Validation Status above.
2. **Well-mixed counterfactuals**: Defining "pure payoffs" in a well-mixed setting is implemented using expected payoffs against the global population ratio. Well-mixed results now match the paper closely, so this choice is not currently a suspected source of error.
3. **Imitation selection**: In the lattice model, if multiple neighbors use the opposite strategy, the agent imitates the neighbor with the highest average payoff. **Tested and ruled out** as a cause of the open lattice-CG gap — with binary strategies, any opposite-strategy neighbor's strategy is just the flip of the agent's own, so this choice has no effect on the outcome.
4. **Bernoulli thinning**: The learning update ratio (η) is modeled as independent probability updates per step (Bernoulli trials). **Tested against exact-count sampling** as an alternative — moved the lattice-CG result by less than 1 percentage point, not the cause of the open gap either.

The lattice-CG discrepancy documented above is real and still unexplained by either of these two flagged choices — see Validation Status for what was tried and what's still open.

---

## Scope of this repository

This repo is the paper replication only — it stays faithful to Tang, Wang & Xing (2025) and should not accumulate product-specific features (reputation systems, punishment mechanisms, custom payoff structures, etc.). That extension work lives in a separate repository, [agent-cooperation-diagnostic](https://github.com/CiMillan/agent-cooperation-diagnostic), which forks this codebase as its starting engine and diverges from here.
