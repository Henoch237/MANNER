"""Rhythmic (periodic) DMP runtime: loads a learned .npy primitive and
integrates it. Used by the executor's stir skill; the recorder in
dmp_assistant/tools/record_dmp.py writes files in this schema."""
import os

import numpy as np


class RhythmicDMP:
    def __init__(self):
        pass

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"DMP file not found: {filepath}")
        data = np.load(filepath, allow_pickle=True).item()
        self.n_dims = data['n_dims']
        self.n_bfs = data['n_bfs']
        self.alpha_z = data['alpha_z']
        self.beta_z = data['beta_z']
        self.c = data['c']
        self.h = data['h']
        self.w = data['w']
        self.g = data['g']
        self.r = data['r']
        self.tau = data['tau']
        self.omega = data['omega']
        self._y = None
        self._z = None

    def _psi(self, theta):
        diff = np.minimum(np.abs(theta - self.c), 2 * np.pi - np.abs(theta - self.c))
        return np.exp(-self.h * diff ** 2)

    def reset(self, y0=None):
        if y0 is not None:
            self._y = np.array(y0, dtype=float).copy()
        else:
            self._y = self.g.copy()
        self._z = np.zeros(self.n_dims)

    def step(self, theta, dt, scale=1.0, goal_offset=None):
        if self._y is None:
            self.reset()
        g = self.g.copy()
        if goal_offset is not None:
            g += np.array(goal_offset)
        tau = self.tau
        psi_vals = self._psi(theta)
        psi_sum = np.sum(psi_vals) + 1e-10
        f = np.zeros(self.n_dims)
        for d in range(self.n_dims):
            f[d] = np.dot(self.w[d, :], psi_vals / psi_sum) * self.r[d] * scale
        for d in range(self.n_dims):
            dz = (self.alpha_z * (self.beta_z * (g[d] - self._y[d]) - self._z[d]) + f[d]) / tau
            dy = self._z[d] / tau
            self._z[d] += dz * dt
            self._y[d] += dy * dt
        return self._y.copy(), self._z.copy() / tau

    def generate(self, t_eval, omega=None, scale=1.0, goal_offset=None, y0=None):
        self.reset(y0)
        if omega is None:
            omega = self.omega
        n_steps = len(t_eval)
        traj = np.zeros((n_steps, self.n_dims))
        vel = np.zeros((n_steps, self.n_dims))
        dt = np.mean(np.diff(t_eval))
        for i in range(n_steps):
            theta = omega * t_eval[i]
            y, dy = self.step(theta, dt, scale=scale, goal_offset=goal_offset)
            traj[i] = y
            vel[i] = dy
        return traj, vel
