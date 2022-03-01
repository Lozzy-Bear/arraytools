#!/bin/python3
# Author: Adam Lozinsky
# Date: June 18, 2019
# Name: Drunk Stumble
# Description: Generates optimal antenna array designs based on
#              perturbed realeaux triangles (Keto, 1997).
import numpy as np


class DrunkenStumble:
    def __init__(self, antennas, wavelength, boundary, weights,
                 elasticity=2.0, dampening=0.1, uniformity=0.89, minsep=1.5, limit=10000):
        # boundary is [xmin, ymin] [xmax, ymax]
        self.num_antennas = len(antennas)
        self.antennas = antennas
        self.weights = weights
        self.wavelength = wavelength
        self.boundary_min = np.ones_like(antennas) * boundary[0, :]
        self.boundary_max = np.ones_like(antennas) * boundary[1, :]
        self.condition = np.ones_like(antennas, dtype=bool)
        self.elasticity = elasticity
        self.dampening = dampening
        self.uniformity = uniformity
        self.minsep = minsep
        self.limit = limit
        self.flag = True

        self.uv_baselines()
        self.run()

    def run(self):
        count = 0
        while self.flag and (count < self.limit):
            self.limit += 1
            self.perturbate()
            self.evaluate()
            self.weighting()
        return

    def perturbate(self):
        self.antennas = np.random.uniform(self.antennas - self.weight, self.antennas + self.weight)
        return

    def evaluate(self):
        self.uv_baselines()
        self.uniqueness()
        for i in range(len(self.baselines)):
            for j in range(len(i + 1, self.baselines)):
                self.separation()
                self.uniqueness()
                self.uniformity()
        if not np.alltrue(self.condition):
            self.flag = False

    def weighting(self):
        self.weights = np.where(self.condition, self.weights * self.elasticity, self.weights * self.dampening)
        self.weights = np.where(self.antennas + self.weights > self.boundary_max, self.boundary_max, self.weights)
        self.weights = np.where(self.antennas - self.weights < self.boundary_min, self.boundary_min, self.weights)
        return

    def uv_baselines(self):
        self.baselines = np.zeros((int((self.num_antennas * (self.num_antennas -2))/2), 2))
        for i in range(self.num_antennas):
            for j in range(i + 1, self.num_antennas):
                self.baselines[i, j] = (self.antennas[i, :] - self.antennas[j, :]) / self.wavelength
                self.baselines[i, j] = (- self.antennas[i, :] + self.antennas[j, :]) / self.wavelength
                u[i, j] = np.append(u, (ant_posx[i] - ant_posx[j]) / self.wavelength)
                v = np.append(v, (ant_posy[i] - ant_posy[j]) / WAVELENGTH)
                u = np.append(u, (ant_posx[j] - ant_posx[i]) / WAVELENGTH)
                v = np.append(v, (ant_posy[j] - ant_posy[i]) / WAVELENGTH)
        self.baselines = np.append(self.baselines, np.conj(self.baselines))
        return

    def separation(self, a, b):
        d = np.sqrt(np.sum((a - b) ** 2, axis=1))
        if d > self.minsep:
            return np.array([True, True])
        else:
            return np.array([False, False])

    def uniqueness(self):

        b = np.array([])
        for i in range(NUM_ANTENNAS):
            for j in range(i + 1, NUM_ANTENNAS):
                d = ((ant_posx[i] - ant_posx[j]) ** 2 + (ant_posy[i] - ant_posy[j]) ** 2) ** 0.5
                b = np.append(b, d)
        ub = len(b)
        for i in range(NUM_ANTENNAS):
            for j in range(i + 1, NUM_ANTENNAS):
                if b[i] <= b[j] + 0.5 and b[i] >= b[j] - 0.5:
                    ub = ub - 1
        if ub >= UNIQUE_BASELINES_REQ:
            return True
        else:
            print("FAILED: Not enough unique baselines.")
            return False

    def uniformity(self):
        self.condition =
        save_index = np.array([], dtype=int)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                r = ((u[i] - u[j]) ** 2 + (v[i] - v[j]) ** 2) ** 0.5
                if r < UNIFROM_RADIUS:
                    save_index = np.append(save_index, i)
                    save_index = np.append(save_index, j)
        mod_x_condition = np.array([False] * NUM_ANTENNAS)
        mod_y_condition = np.array([False] * NUM_ANTENNAS)
        for i in save_index:
            mod_x_condition[si[i, 0]] = True
            mod_x_condition[si[i, 1]] = True
            mod_y_condition[si[i, 0]] = True
            mod_y_condition[si[i, 1]] = True
        if not any(mod_x_condition) and not any(mod_y_condition):
            condition = True
        else:
            condition = False
        return condition, mod_x_condition, mod_y_condition


if __name__ == "__main__":
    ds = DrunkenStumble(np.array([[1.0, 3.0], [0.0, 0.0]]), 6.0, np.array([[0.0, 0.0], [10.0, 10.0]]),
                        np.array([[1,1], [1,1]]))


