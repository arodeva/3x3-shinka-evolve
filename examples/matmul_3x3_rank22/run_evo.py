#!/usr/bin/env python3
import argparse

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

task_sys_msg = """You are an expert in algebraic complexity theory and numerical optimization. The task is to find a scheme of rank 22 for 3x3 matrix multiplication: factor matrices U, V, W (each 9x22) satisfying the Brent equations sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] exactly, ideally with entries in {-1, 0, 1}.

Be realistic about this target: no rank-22 scheme for general 3x3 matrix multiplication is known to exist. Rank 23 (Laderman, 1976) has stood as the best known upper bound for decades, and dedicated searches using flip-graph exploration and SAT solvers have specifically looked for rank 22 and not found one. Treat this as a long-shot exploratory search, not a near-certain win. A program that reliably gets very close (tiny residual after rounding) is valuable progress even if it never reaches exact.

You evolve the SEARCH ALGORITHM (find_scheme), not a hard-coded table of coefficients. It is judged on several fixed seeds under a wall-clock budget, so robust, fast search beats a single lucky run. Rank 22 has one fewer degree of freedom than rank 23, so the manifold of valid decompositions is tighter -- expect continuous methods (ALS, gradient descent) to struggle more to even approximately fit, let alone land on an exact integer solution.

Directions to explore:
1. ALS with regularization schedules, Levenberg-Marquardt, or gradient descent with Adam
2. Penalties that push entries toward {-1, 0, 1} (annealed), then rounding and repair
3. Fixing entries (sparsity / discretization) and re-solving the rest by least squares
4. Exploiting symmetries (cyclic Z3 symmetry, sandwiching invariance) to reduce the search space
5. Smarter initialization and restart policies; discrete local search on rounded schemes
6. Since exact solutions are unlikely, also consider: a scoring-aware search that spends effort
   finding and reporting the closest approximate scheme, so progress is visible even without
   an exact hit

Do not depend on external data files. Use only numpy/scipy."""


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = task_sys_msg
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py", time="00:10:00")
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_smoke.yaml")
    args = parser.parse_args()
    main(args.config_path)
