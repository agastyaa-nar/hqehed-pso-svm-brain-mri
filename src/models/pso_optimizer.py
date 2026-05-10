"""
pso_optimizer.py — Particle Swarm Optimization for SVM hyperparameter tuning.

Optimizes (C, gamma) of SVM-RBF kernel using Dice coefficient as fitness
function (Section 3.5, Eq. 2.27–2.28).
"""

from __future__ import annotations
from typing import Tuple, List

import numpy as np
from sklearn.svm import SVC

from ..core.config import Config


class PSOOptimizer:
    """PSO-based optimizer for SVM hyperparameters C and gamma.

    Search is conducted in log10 space:
        position[0] = log10(C)   in cfg.pso_c_range
        position[1] = log10(g)   in cfg.pso_g_range

    Fitness = Dice coefficient on validation set.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cfg: Config,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cfg = cfg

        self.bounds = np.array([cfg.pso_c_range, cfg.pso_g_range])
        self.pos = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(cfg.pso_particles, 2)
        )
        self.vel = np.zeros_like(self.pos)
        self.pbest = self.pos.copy()
        self.pbest_fit = np.full(cfg.pso_particles, -np.inf)
        self.gbest: np.ndarray = self.pos[0].copy()
        self.gbest_fit: float = -np.inf
        self.history: List[float] = []

    def _fitness(self, pos_row: np.ndarray) -> float:
        C = 10.0 ** pos_row[0]
        g = 10.0 ** pos_row[1]
        try:
            clf = SVC(C=C, gamma=g, kernel="rbf", class_weight="balanced")
            clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_val)
            tp = int(np.sum((pred == 1) & (self.y_val == 1)))
            fp = int(np.sum((pred == 1) & (self.y_val == 0)))
            fn = int(np.sum((pred == 0) & (self.y_val == 1)))
            denom = 2 * tp + fp + fn
            return float(2 * tp / denom) if denom > 0 else 0.0
        except Exception:
            return 0.0

    def optimize(self, verbose: bool = True) -> Tuple[float, float, List[float]]:
        """Run PSO and return (C_opt, g_opt, convergence_history)."""
        cfg = self.cfg

        # Initialize personal bests
        for i in range(cfg.pso_particles):
            fit = self._fitness(self.pos[i])
            self.pbest_fit[i] = fit
            if fit > self.gbest_fit:
                self.gbest_fit = fit
                self.gbest = self.pos[i].copy()

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  PSO: {cfg.pso_particles} particles  |  {cfg.pso_iterations} iterations")
            print(f"{'─'*55}")

        for t in range(cfg.pso_iterations):
            r1 = np.random.rand(cfg.pso_particles, 2)
            r2 = np.random.rand(cfg.pso_particles, 2)
            self.vel = (
                cfg.pso_w * self.vel
                + cfg.pso_c1 * r1 * (self.pbest - self.pos)
                + cfg.pso_c2 * r2 * (self.gbest - self.pos)
            )
            self.pos += self.vel
            self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])

            for i in range(cfg.pso_particles):
                fit = self._fitness(self.pos[i])
                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest[i] = self.pos[i].copy()
                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest = self.pos[i].copy()

            self.history.append(self.gbest_fit)
            if verbose and (t + 1) % 10 == 0:
                print(
                    f"  Iter {t+1:3d}/{cfg.pso_iterations}  "
                    f"gbest Dice={self.gbest_fit:.4f}  "
                    f"C={10.**self.gbest[0]:.4f}  "
                    f"γ={10.**self.gbest[1]:.6f}"
                )

        C_opt = float(10.0 ** self.gbest[0])
        g_opt = float(10.0 ** self.gbest[1])
        return C_opt, g_opt, self.history
