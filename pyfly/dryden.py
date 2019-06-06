import numpy as np
from scipy.signal import lti
import math


class DrydenGustModel:
    def __init__(self, size, dt, b, h=100, V_a=25, intensity=None):
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

        # Convert back to m/s in the numerator (NB: this should not carry over to angular rates below)
        self.H_u = lti(feet2meters * K_u, [T_u, 1])
        self.H_v = lti([feet2meters * K_v * T_v1, feet2meters * K_v], [T_v2 ** 2, 2 * T_v2, 1])
        self.H_w = lti([feet2meters * K_w * T_w1, feet2meters * K_w], [T_w2 ** 2, 2 * T_w2, 1])

        K_p = sigma_w * math.sqrt(0.8 / V_a) * ((math.pi / (4 * b)) ** (1 / 6)) / ((L_w) ** (1 / 3))
        K_q = 1 / V_a
        K_r = K_q

        T_p = 4 * b / (math.pi * V_a)
        T_q = T_p
        T_r = 3 * b / (math.pi * V_a)

        self.H_p = lti(K_p, [T_p, 1])
        self.H_q = lti([-K_w * K_q * T_w1, -K_w * K_q, 0], [T_q * T_w2 ** 2, T_w2 ** 2 + 2 * T_q * T_w2, T_q + 2 * T_w2, 1])
        self.H_r = lti([K_v * K_r * T_v1, K_v * K_r, 0], [T_r * T_v2 ** 2, T_v2 ** 2 + 2 * T_r * T_v2, T_r + 2 * T_v2, 1])

        self.np_random = None
        self.seed()

        self.dt = dt
        self.sim_length = None

        #self.noise = None
        self.vel_lin = None
        self.vel_ang = None

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def generate_noise(self, size):
        noise = [math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   math.sqrt(math.pi / self.dt) * self.np_random.standard_normal(size=size),
                   ]

        return noise

    def reset(self):
        self.vel_lin = None
        self.vel_ang = None
        self.sim_length = 0

    def simulate(self, length, noise=None):
        t_span = [self.sim_length, self.sim_length + length]

        t = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) / self.dt)

        if noise is None:
            noise = self.generate_noise(t.shape[0])

        vel_lin = np.array([self.H_u.output(noise[0], t)[1],
                                   self.H_v.output(noise[1], t)[1],
                                   self.H_w.output(noise[2], t)[1]])

        vel_ang = np.array([self.H_p.output(noise[3], t)[1],
                                     self.H_q.output(noise[3], t)[1],
                                     self.H_r.output(noise[3], t)[1]])

        if self.vel_lin is None:
            self.vel_lin = vel_lin
            self.vel_ang = vel_ang
        else:
            self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)
            self.vel_ang = np.concatenate((self.vel_ang, vel_ang), axis=1)

        self.sim_length += length