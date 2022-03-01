# Author: Adam Lozinsky
# Date: June 17, 2019
# Name: Jacobs Ralston
# Description: Use 'Ambiguity Resolution in Interferometry (Jacobs,. Ralston, 1981)'
#              To determine the best antenna spacing in a crossed configuration.
import numpy as np


class JacobsRalston:
    def __init__(self, antennas, end, iterations, step=0.001, minsep=1.5):
        """
        For two antennas (antenna and end) this algorithm determines to optimal
        phase error minimization point to place a third antenna between the
        first two. This is then repeated as for as many (iterations) as
        required. A final average optimal point can be found. This process
        should be repeated for all possible combination of antennas along
        a linear array. Qualitative analysis must be used in conjunction.

        Parameters
        ----------
        antennas : float np.array
            [a1, a2, ...] the first antenna and any assumed or required
            antenna locations (<=iterations) in [lambda] from a1.
        end : float
            the last antenna location in [lambda] from a1.
        iterations : int
            number of antennas to place between a1 and end.
        step : float
            step size of the calculation, default = 0.1*lambda.
        minsep : float
            minimum distance two antennas can be placed in [lambda].
        """
        self.antennas = antennas
        self.end = end
        self.x = np.arange(antennas[0] + minsep, end - minsep, step)
        self.y = np.ones((iterations, self.x.shape[0]))
        self.step = step
        self.minsep = minsep
        self.iterations = iterations

        self.run()
        self.y_total = np.sum(self.y, axis=0) / len(self.y)
        self.antennas = np.append(self.antennas, self.end)

    def run(self):
        for i in range(self.iterations):
            self.y[i, :], pos = self.find_position(self.antennas[i], self.end)
            if i <= self.antennas.shape[0]:
                self.antennas = np.append(self.antennas, pos)
        return

    def find_position(self, x1, x2):
        """Find the next position for the antenna in the linear array. """
        x = np.arange(x1 + self.minsep, x2 - self.minsep, self.step)
        y = np.zeros_like(x)
        n1 = self.modulo_integer(np.abs(x2 - x1) / 2, np.pi / 2)
        for i in range(len(x)):
            n2 = self.modulo_integer(x[i], np.pi / 2)
            y[i] = 1 + self.find_lmin(np.abs(x2 - x1) / 2, x[i], n1, n2)
        idx = np.argmax(y)
        y = self.pad_like(y, self.x, value=1.0)
        return y, x[idx]

    @staticmethod
    def pad_like(arr, ref, value=0.0, mode='before'):
        """Pad the array either before or after with any value."""
        """Pad array given a reference"""
        pad = np.ones_like(ref) * value
        if mode is 'before':
            pad[ref.shape[0] - arr.shape[0]:] = arr
        elif mode is 'after':
            pad[:arr.shape[0]] = arr
        return pad

    @staticmethod
    def modulo_integer(d, theta_max):
        """Find the largest n (Jacobs and Ralston, 1981)."""
        nmax = int(np.abs(d * np.sin(theta_max) + 0.5))
        n = np.arange(-nmax, nmax, 1, dtype='int')
        return n

    @staticmethod
    def find_lmin(d1, d2, n1, n2):
        """Find the smallest l (Jacobs and Ralston, 1981)."""
        y = np.zeros((len(n1), len(n2)))
        for i in range(len(n1)):
            for j in range(len(n2)):
                y[i, j] = d1 / d2 * n2[j] - n1[i]
        y = np.sort(y.flatten())
        lmin = np.min((y[1::] - y[0:-1]) / ((1 + (d1 / d2) ** 2) ** 0.5))
        return lmin


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    jr = JacobsRalston(np.array([0.0]), 16.2, 3)
    print('antennas:', jr.antennas)

    plt.figure(figsize=[6, 4])
    plt.plot(jr.x, jr.y[0, :], color='orange', label='Between Antenna 1 and 2')
    plt.title("Minimum Separation vs. Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Minimum Phase Lines Separation")
    plt.legend(loc='upper right')
    plt.savefig('jr_minsep_12.pdf')

    plt.figure(figsize=[6, 4])
    plt.plot(jr.x, jr.y[1, :], 'm', label='Between Antenna 1 and 3')
    plt.title("Minimum Separation vs. Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Minimum Phase Lines Separation")
    plt.legend(loc='upper right')
    plt.savefig('jr_minsep_13.pdf')

    plt.figure(figsize=[6, 4])
    plt.plot(jr.x, jr.y[2, :], 'g', label='Between Antenna 1 and 4')
    plt.title("Minimum Separation vs. Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Minimum Phase Lines Separation")
    plt.legend(loc='upper right')
    plt.savefig('jr_minsep_14.pdf')

    plt.figure(figsize=[6, 4])
    plt.plot(jr.x, jr.y_total, 'k', label='Averaged')
    plt.title("Minimum Separation vs. Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Minimum Phase Lines Separation")
    plt.legend(loc='upper right')
    plt.savefig('jr_minsep_combine.pdf')

    plt.figure(figsize=[6, 4])
    plt.plot(jr.x, jr.y_total, 'k', label='Averaged')
    plt.title("Minimum Separation vs. Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Minimum Phase Lines Separation")
    plt.legend(loc='upper right')
    plt.xlim(8.7, 9.3)
    plt.savefig('jr_minsep_combine_zoom.pdf')

    plt.show()
