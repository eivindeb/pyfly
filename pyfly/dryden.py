import numpy as np
from scipy.signal import lti
import math


class Filter:
    def __init__(self, num, den):
        """
        Wrapper for the scipy LTI system class.
        :param num: numerator of transfer function
        :param den: denominator of transfer function
        """
        self.filter = lti(num, den)
        self.x = None
        self.y = None
        self.t = None

    def simulate(self, u, t):
        """
        Simulate filter
        :param u: filter input
        :param t: time steps for which to simulate
        :return: filter output
        """
        if self.x is None:
            x_0 = None
        else:
            x_0 = self.x[-1]

        self.t, self.y, self.x = self.filter.output(U=u, T=t, X0=x_0)

        return self.y

    def reset(self):
        """
        Reset filter
        :return:
        """
        self.x = None
        self.y = None
        self.t = None


class DrydenGustModel:
    def __init__(self, dt, b, h=100, V_a=25, intensity=None):
        """
        Python realization of the continuous Dryden Turbulence Model (MIL-F-8785C).
        :param dt: (float) band-limited white noise input sampling time.
        :param b: (float) wingspan of aircraft
        :param h: (float) Altitude of aircraft
        :param V_a: (float) Airspeed of aircraft
        :param intensity: (str) Intensity of turbulence, one of ["light", "moderate", "severe"]
        """
        # For fixed (nominal) altitude and airspeed
        h = h  # altitude [m]
        V_a = V_a  # airspeed [m/s]

        # Conversion factors
        # 1 meter = 3.281 feet
        meters2feet = 3.281
        feet2meters = 1 / meters2feet
        # 1 knot = 0.5144 m/s
        knots2mpers = 0.5144

        if intensity is None:
            W_20 = 15 * knots2mpers  # light turbulence
        elif intensity == "light":
            W_20 = 15 * knots2mpers  # light turbulence
        elif intensity == "moderate":
            W_20 = 30 * knots2mpers  # moderate turbulence
        elif intensity == "severe":
            W_20 = 45 * knots2mpers  # severe turbulence
        else:
            raise Exception("Unsupported intensity type")

        # Convert meters to feet and follow MIL-F-8785C spec
        h = h * meters2feet
        b = b * meters2feet
        V_a = V_a * meters2feet
        W_20 = W_20 * meters2feet

        # Turbulence intensities
        sigma_w = 0.1 * W_20
        sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        sigma_v = sigma_u

        # Turbulence length scales
        L_u = h / (0.177 + 0.000823 * h) ** 1.2
        L_v = L_u
        L_w = h

        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V_a))
        K_v = sigma_v * math.sqrt((L_v) / (math.pi * V_a))
        K_w = sigma_w * math.sqrt((L_w) / (math.pi * V_a))

        T_u = L_u / V_a
        T_v1 = math.sqrt(3.0) * L_v / V_a
        T_v2 = L_v / V_a
        T_w1 = math.sqrt(3.0) * L_w / V_a
        T_w2 = L_w / V_a

        K_p = sigma_w * math.sqrt(0.8 / V_a) * ((math.pi / (4 * b)) ** (1 / 6)) / ((L_w) ** (1 / 3))
        K_q = 1 / V_a
        K_r = K_q

        T_p = 4 * b / (math.pi * V_a)
        T_q = T_p
        T_r = 3 * b / (math.pi * V_a)

        self.filters = {"H_u": Filter(feet2meters * K_u, [T_u, 1]),
                        "H_v": Filter([feet2meters * K_v * T_v1, feet2meters * K_v], [T_v2 ** 2, 2 * T_v2, 1]),
                        "H_w": Filter([feet2meters * K_w * T_w1, feet2meters * K_w], [T_w2 ** 2, 2 * T_w2, 1]),
                        "H_p": Filter(K_p, [T_p, 1]),
                        "H_q": Filter([-K_w * K_q * T_w1, -K_w * K_q, 0], [T_q * T_w2 ** 2, T_w2 ** 2 + 2 * T_q * T_w2, T_q + 2 * T_w2, 1]),
                        "H_r": Filter([K_v * K_r * T_v1, K_v * K_r, 0], [T_r * T_v2 ** 2, T_v2 ** 2 + 2 * T_r * T_v2, T_r + 2 * T_v2, 1]),}

        self.np_random = None
        self.seed()

        self.dt = dt
        self.sim_length = None

        self.noise = None

        self.vel_lin = None
        self.vel_ang = None

    def seed(self, seed=None):
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)

    def _generate_noise(self, size):
        return np.sqrt(np.pi / self.dt) * self.np_random.standard_normal(size=(4, size))

    def reset(self, noise=None):
        """
        Reset model.
        :param noise: (np.array) Input to filters, should be four sequences of Gaussianly distributed numbers.
        :return:
        """
        self.vel_lin = None
        self.vel_ang = None
        self.sim_length = 0

        if noise is not None:
            assert len(noise.shape) == 2
            assert noise.shape[0] == 4
            noise = noise * math.sqrt(math.pi / self.dt)
        self.noise = noise

        for filter in self.filters.values():
            filter.reset()

    def simulate(self, length):
        """
        Simulate turbulence by passing band-limited Gaussian noise of length length through the shaping filters.
        :param length: (int) the number of steps to simulate.
        :return:
        """
        t_span = [self.sim_length, self.sim_length + length]

        t = np.linspace(t_span[0] * self.dt, t_span[1] * self.dt, length)

        if self.noise is None:
            noise = self._generate_noise(t.shape[0])
        else:
            if self.noise.shape[-1] >= t_span[1]:
                noise = self.noise[:, t_span[0]:t_span[1]]
            else:
                noise_start_i = t_span[0] % self.noise.shape[-1]
                remaining_noise_length = self.noise.shape[-1] - noise_start_i
                if remaining_noise_length >= length:
                    noise = self.noise[:, noise_start_i:noise_start_i + length]
                else:
                    if length - remaining_noise_length > self.noise.shape[-1]:
                        concat_noise = np.pad(self.noise,
                                              ((0, 0), (0, length - remaining_noise_length - self.noise.shape[-1])),
                                              mode="wrap")
                    else:
                        concat_noise = self.noise[:, :length - remaining_noise_length]
                    noise = np.concatenate((self.noise[:, noise_start_i:], concat_noise), axis=-1)

        vel_lin = np.array([self.filters["H_u"].simulate(noise[0], t),
                            self.filters["H_v"].simulate(noise[1], t),
                            self.filters["H_w"].simulate(noise[2], t)])

        vel_ang = np.array([self.filters["H_p"].simulate(noise[3], t),
                            self.filters["H_q"].simulate(noise[1], t),
                            self.filters["H_r"].simulate(noise[2], t)])

        if self.vel_lin is None:
            self.vel_lin = vel_lin
            self.vel_ang = vel_ang
        else:
            self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)
            self.vel_ang = np.concatenate((self.vel_ang, vel_ang), axis=1)

        self.sim_length += length