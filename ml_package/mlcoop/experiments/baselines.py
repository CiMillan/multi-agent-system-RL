from __future__ import annotations
from mlcoop.core.config import PopulationConfig, RLParams, ILParams, MLParams
from mlcoop.core.simulators import simulate_well_mixed, simulate_square_lattice


def baseline_params_well_mixed(seed=42):
    # Paper baseline for well-mixed ML experiments uses N=10000, maxgen=100,
    # FC0=FD0=3, H0=3, and reported parameters gamma=0.5, alpha=0.3, delta=0.01,
    # rho=0.01, phi=0.8 (page 7). alpha (introspection rate) maps to the IL
    # module's eta; RLParams has no alpha field of its own.
    return {
        "pop_cfg": PopulationConfig(N=10000, initial_cooperator_fraction=0.5, seed=seed),
        "rl_params": RLParams(phi=0.8, rho=0.01, delta=0.01, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0),
        "il_params": ILParams(eta=0.3),
        "ml_params": MLParams(theta=0.5, max_steps=100),
    }


def baseline_params_lattice(seed=42):
    # Paper baseline for the square-lattice ML experiments uses the same
    # gamma=0.5, alpha=0.3, delta=0.01, rho=0.01, phi=0.8 (page 9), but
    # maxgen=200 and a 100x100 grid (N=10000).
    return {
        "pop_cfg": PopulationConfig(N=10000, initial_cooperator_fraction=0.5, seed=seed),
        "rl_params": RLParams(phi=0.8, rho=0.01, delta=0.01, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0),
        "il_params": ILParams(eta=0.3),
        "ml_params": MLParams(theta=0.5, max_steps=200),
    }


def run_well_mixed_baseline(game_name, seed=42):
    cfg = baseline_params_well_mixed(seed=seed)
    return simulate_well_mixed(game_name, **cfg)


def run_square_lattice_baseline(game_name, seed=42):
    cfg = baseline_params_lattice(seed=seed)
    return simulate_square_lattice(game_name, rows=100, cols=100, **cfg)


if __name__ == "__main__":
    for game in ["PDG", "CG", "CoG"]:
        result = run_well_mixed_baseline(game, seed=42)
        print(game, "well-mixed final MFC:", round(float(result["mfc"][-1]), 4))
