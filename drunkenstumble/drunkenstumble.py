# Author: Adam Lozinsky
# Date: June 18, 2019
# Name: Drunk Stumble
# Description: Generates optimal antenna array designs based on
#              perturbed realeaux triangles (Keto, 1997).
import numpy as np


class DrunkenStumble:
    def __init__(self, antennas, wavelength, boundary, weights=np.array([]),
                 elasticity=2.0, dampening=0.1, uniformity=0.89, minsep=1.5, limit=10000):
        """
        Drunken Stumble is a simulated annealing like antenna array optimization algorithm. It attempts to find
        optimal array patterns where; the arrays are bounded, have minimal closure distance to avoid mutual coupling
        effects, desire all unique baselines, and uniformity of the uv-plane. The inspiration behind the algorithm is
        the chaotic behaviour a chorus of drunks display while trying to walk home together.

        Parameters
        ----------
        antennas : float np.array
            [[x1, y1], [x2, y2], ...] positions of the antennas in local coordinates [meters].
        wavelength : float
            array centered wavelength.
        boundary : float np.array
            [[xmin, xmax], [ymin, ymax]] the bounded region the antennas can be placed within.
        weights : float np.array
            [[w1, w1], [w2, w2], ...] the same shape as antennas, this holds the initial weights each antenna can be
            perturbed by. Set this to 0.0 to keep an antenna from ever moving.
        elasticity : float
            the amount of energy to add to an antennas if it is not placed well, w *= e.
        dampening : float
            the amount to reduce an antennas energy by if finds a local or global minima, w *= d.
        uniformity : float
            the maximum allowable distance two visibility points (location not value) can be next to each other.
        minsep : float
            the minimum distance two antenna can be placed next to each other in units of lambda.
        limit : int
            maximum number of iteration to preform per simulation.
        """
        if not weights:
            self.weights = np.ones_like(antennas) * 2.0
        self.num_antennas = len(antennas)
        self.unique_baselines = int(self.num_antennas * (self.num_antennas - 1)/2)
        self.antennas = antennas
        self.baselines = np.zeros((self.unique_baselines, 2))
        self.order = np.zeros_like(self.baselines, dtype=int)
        self.wavelength = wavelength
        self.boundary_min = np.ones_like(antennas) * boundary[0, :]
        self.boundary_max = np.ones_like(antennas) * boundary[1, :]
        self.weights_maximum = np.abs(self.boundary_min - self.boundary_max)
        self.condition = np.ones_like(antennas, dtype=np.bool)
        self.elasticity = elasticity
        self.dampening = dampening
        self.uniformity = uniformity
        self.minsep = minsep
        self.limit = limit
        self.flag = True

        self.run()

    def run(self):
        count = 0
        while self.flag and (count < self.limit):
            print(f"\rsimulating iteration:\t{count:05d}/{self.limit}", end='')
            count += 1
            self.perturbate()
            self.evaluate()
            self.weighting()
        self.baselines = np.concatenate((self.baselines, -1*self.baselines), axis=0)
        print('\tdone')
        return

    def perturbate(self):
        pmax = np.where(self.antennas + self.weights > self.boundary_max,
                        self.boundary_max,
                        self.antennas - self.weights)
        pmin = np.where(self.antennas - self.weights < self.boundary_min,
                        self.boundary_min,
                        self.antennas + self.weights)
        self.antennas = np.random.uniform(pmin, pmax)
        return

    def evaluate(self):
        self.condition *= False
        count = 0
        for i in range(self.num_antennas):
            for j in range(i + 1, self.num_antennas):
                self.baselines[count, :] = (self.antennas[i, :] - self.antennas[j, :]) / self.wavelength
                self.order[count, :] = np.array([i, j])
                # Check for minimal separation
                d = np.sqrt(self.baselines[count, 0] ** 2 + self.baselines[count, 1] ** 2)
                if d <= self.minsep:
                    self.condition[i, :] = np.array([True, True])
                    self.condition[j, :] = np.array([True, True])
                count += 1
        rounded_baselines = np.copy(np.round(self.baselines, decimals=2))
        for m in range(len(self.baselines)):
            # Check for uniqueness
            if (rounded_baselines[m, :] in rounded_baselines[:m, :]) or\
                    (rounded_baselines[m, :] in rounded_baselines[m+1:, :]):
                self.condition[self.order[m, 0], :] = np.array([True, True])
                self.condition[self.order[m, 1], :] = np.array([True, True])
            for n in range(m + 1, len(self.baselines)):
                # Check for uniformity
                d = np.sqrt(np.sum((self.baselines[m, :] - self.baselines[n, :]) ** 2))
                if d < self.uniformity:
                    self.condition[self.order[m, 0], :] = np.array([True, True])
                    self.condition[self.order[m, 1], :] = np.array([True, True])
                    self.condition[self.order[n, 0], :] = np.array([True, True])
                    self.condition[self.order[n, 1], :] = np.array([True, True])
        if not np.alltrue(self.condition):
            self.flag = False

    def weighting(self):
        self.weights = np.where(self.condition, self.weights * self.elasticity, self.weights * self.dampening)
        self.weights = np.where(self.antennas + self.weights > self.boundary_max, self.weights_maximum, self.weights)
        self.weights = np.where(self.antennas - self.weights < self.boundary_min, self.weights_maximum, self.weights)
        return


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ants = np.random.uniform(5, 15, (10, 2))
    bounds = np.array([[0.0, 0.0], [200.0, 200.0]])
    ds = DrunkenStumble(ants, 6.0, bounds)
    print('antennas:\n', ds.antennas)
    print('baselines:\n', ds.baselines)
    print('weights:\n', ds.weights)

    plt.figure()
    plt.subplot(121)
    plt.scatter(ds.antennas[:, 0], ds.antennas[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('position domain')
    plt.grid()
    plt.subplot(122)
    plt.scatter(ds.baselines[:, 0], ds.baselines[:, 1])
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('visibility domain')
    plt.grid()
    plt.show()
