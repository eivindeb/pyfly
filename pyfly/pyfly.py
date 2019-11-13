import numpy as np
import math
import scipy.integrate
import scipy.io
import os.path as osp
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec


class ConstraintException(Exception):
    def __init__(self, variable, value, limit):
        self.message = "Constraint on {} violated ({}/{})".format(variable, value, limit)
        self.variable = variable


class Variable:
    def __init__(self, name, value_min=None, value_max=None, init_min=None, init_max=None, constraint_min=None,
                 constraint_max=None, convert_to_radians=False, unit=None, label=None, wrap=False):
        """
        PyFly state object managing state history, constraints and visualizations.

        :param name: (string) name of state
        :param value_min: (float) lowest possible value of state, values will be clipped to this limit
        :param value_max: (float) highest possible value of state, values will be clipped to this limit
        :param init_min: (float) lowest possible initial value of state
        :param init_max: (float) highest possible initial value of state
        :param constraint_min: (float) lower constraint of state, which if violated will raise ConstraintException
        :param constraint_max: (float) upper constraint of state, which if violated will raise ConstraintException
        :param convert_to_radians: (bool) whether to convert values for attributes from configuration file from degrees
        to radians
        :param unit: (string) unit of the state, for plotting purposes
        :param label: (string) label given to state in plots
        :param wrap: (bool) whether to wrap state value in region [-pi, pi]
        """
        self.value_min = value_min
        self.value_max = value_max

        self.init_min = init_min if init_min is not None else value_min
        self.init_max = init_max if init_max is not None else value_max

        self.constraint_min = constraint_min
        self.constraint_max = constraint_max

        if convert_to_radians:
            for attr_name, val in self.__dict__.items():
                if val is not None:
                    setattr(self, attr_name, np.radians(val))

        self.name = name

        self.value = None

        self.wrap = wrap

        self.unit = unit
        self.label = label if label is not None else self.name
        self.lines = {"self": None}
        self.target_lines = {"self": None}
        self.target_bounds = {"self": None}

        self.np_random = None
        self.seed()

        self.history = None

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (float) initial value of state
        """
        self.history = []

        if value is None:
            try:
                value = self.np_random.uniform(self.init_min, self.init_max)
            except TypeError:
                raise Exception("Variable init_min and init_max can not be None if no value is provided on reset")
        else:
            value = self.apply_conditions(value)

        self.value = value

        self.history.append(value)

    def seed(self, seed=None):
        """
        Seed random number generator of state

        :param seed: (int) seed of random state
        """
        self.np_random = np.random.RandomState(seed)

    def apply_conditions(self, value):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints are applied
        :return: (float) value after applying limits and constraints
        """
        if self.constraint_min is not None and value < self.constraint_min:
            raise ConstraintException(self.name, value, self.constraint_min)

        if self.constraint_max is not None and value > self.constraint_max:
            raise ConstraintException(self.name, value, self.constraint_max)

        if self.value_min is not None or self.value_max is not None:
            value = np.clip(value, self.value_min, self.value_max)

        if self.wrap and np.abs(value) > np.pi:
            value = np.sign(value) * (np.abs(value) % np.pi - np.pi)

        return value

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value of state
        :param save: (bool) whether to commit value to history of state
        """
        value = self.apply_conditions(value)

        if save:
            self.history.append(value)

        self.value = value

    def plot(self, axis=None, y_unit=None, target=None, plot_id=None, **plot_kw):
        """
        Plot state history.

        :param axis: (matplotlib.pyplot.axis or None) axis object to plot state to. If None create new axis
        :param y_unit: (string) unit state should be plotted in, will convert values if different from internal
        representation
        :param target: (list) target values for state, must be of equal size to state history
        :param plot_id: (string or int or None) identifier of parent plot object. Allows state to plot to multiple
        figures at a time.
        :param plot_kw: (dict) plot keyword arguments passed to matplotlib.pyplot.plot
        """

        def linear_scaling(val, old_min, old_max, new_min, new_max):
            return (new_max - np.sign(old_min) * (- new_min)) / (old_max - old_min) * (
                        np.array(val) - old_max) + new_max

        if y_unit is None:
            y_unit = self.unit if y_unit is None else y_unit

        x, y = self._get_plot_x_y_data()
        if "degrees" in y_unit:
            y = np.degrees(y)
            if target is not None:
                target["data"] = np.degrees(target["data"])
                if "bound" in target:
                    target["bound"] = np.degrees(target["bound"])
        elif y_unit == "%":  # TODO: scale positive according to positive limit and negative according to lowest minimum value
            y = linear_scaling(y, self.value_min, self.value_max, -100, 100)
            if target is not None:
                target["data"] = linear_scaling(target["data"], self.value_min, self.value_max, -100, 100)
                if "bound" in target:
                    target["bound"] = linear_scaling(target["bound"], self.value_min, self.value_max, -100, 100)
        else:
            y = y

        plot_object = axis
        if axis is None:
            plot_object = plt
            plot_id = "self"
            fig_kw = {"title": self.name, "ylabel": y_unit}

        if self.lines.get(plot_id, None) is None:
            line, = plot_object.plot(x, y, label=self.label, **plot_kw)
            self.lines[plot_id] = line

            if target is not None:
                tar_line, = plot_object.plot(x, target["data"], color=self.lines[plot_id].get_color(), linestyle="dashed",
                                             marker="x", markevery=0.2)

                if "bound" in target:
                    tar_bound = plot_object.fill_between(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"], alpha=0.15,
                                                         facecolor=self.lines[plot_id].get_color()
                                                        )
                    self.target_bounds[plot_id] = tar_bound
                self.target_lines[plot_id] = tar_line
        else:
            self.lines[plot_id].set_data(x, y)
            if target is not None:
                self.target_lines[plot_id].set_data(x, target)
                if "bound" in target:  # TODO: fix this?
                    self.target_bounds[plot_id].set_data(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"])
        if axis is None:
            for k, v in fig_kw.items():
                getattr(plot_object, format(k))(v)
            plt.show()

    def close_plot(self, plot_id="self"):
        """
        Close plot with id plot_id.

        :param plot_id: (string or int) identifier of parent plot object
        """
        self.lines[plot_id] = None
        self.target_lines[plot_id] = None
        self.target_bounds[plot_id] = None

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history.

        :return: ([int], [float]) x plot data, y plot data
        """
        x = list(range(len(self.history)))
        y = self.history
        return x, y


class ControlVariable(Variable):
    def __init__(self, order=None, tau=None, omega_0=None, zeta=None, dot_max=None, disabled=False, **kwargs):
        """
        PyFly actuator state variable.

        :param order: (int) order of state transfer function
        :param tau: (float) time constant for first order transfer functions
        :param omega_0: (float) undamped natural frequency of second order transfer functions
        :param zeta: (float) damping factor of second order transfer function
        :param dot_max: (float) constraint on magnitude of derivative of second order transfer function
        :param disabled: (bool) if actuator is disabled for aircraft, e.g. aircraft has no rudder
        :param kwargs: (dict) keyword arguments for Variable class
        """
        assert (disabled or (order == 1 or order == 2))
        super().__init__(**kwargs)
        self.order = order
        self.tau = tau
        self.omega_0 = omega_0
        self.zeta = zeta
        self.dot_max = dot_max

        if order == 1:
            assert (tau is not None)
            self.coefs = [[-1 / self.tau, 0, 1 / self.tau], [0, 0, 0]]
        elif order == 2:
            assert (omega_0 is not None and zeta is not None)
            self.coefs = [[0, 1, 0], [-self.omega_0 ** 2, -2 * self.zeta * self.omega_0, self.omega_0 ** 2]]
        self.dot = None
        self.command = None
        self.disabled = disabled
        if self.disabled:
            self.value = 0
        self.plot_quantity = "value"

    def apply_conditions(self, values):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints is applied
        :return: (float) value after applying limits and constraints
        """
        try:
            value, dot = values
        except:
            value, dot = values, 0
        value = super().apply_conditions(value)

        if self.dot_max is not None:
            dot = np.clip(dot, -self.dot_max, self.dot_max)

        return [value, dot]

    def set_command(self, command):
        """
        Set setpoint for actuator and commit to history of state

        :param command: setpoint for actuator
        """
        command = super().apply_conditions(command)
        self.command = command
        self.history["command"].append(command)

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (list) initial value, derivative and setpoint of state
        """
        self.history = {"value": [], "dot": [], "command": []}

        if not self.disabled:
            if value is None:
                value = self.np_random.uniform(self.init_min, self.init_max), 0
            else:
                value = self.apply_conditions(value)

            self.value = value[0]
            self.dot = value[1]
            command = None
            self.command = command
        else:
            value, dot, command = 0, 0, None
            self.value = value
            self.dot = dot
            self.command = command

        self.history["value"].append(self.value)
        self.history["dot"].append(self.dot)

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value and derivative of state
        :param save: (bool) whether to commit value to history of state
        """
        value, dot = self.apply_conditions(value)

        self.value = value
        self.dot = dot

        if save:
            self.history["value"].append(value)
            self.history["dot"].append(dot)

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history, for the quantity designated by the attribute plot_quantity.

        :return: ([int], [float]) x plot data, y plot data
        """
        y = self.history[self.plot_quantity]
        x = list(range(len(y)))
        return x, y

    def get_coeffs(self):
        if self.order == 1:
            return
        else:
            return []


class EnergyVariable(Variable):
    def __init__(self, mass=None, inertia_matrix=None, gravity=None, **kwargs):
        super().__init__(**kwargs)
        self.required_variables = []
        self.variables = {}
        if self.name == "energy_potential" or self.name == "energy_total":
            assert(mass is not None and gravity is not None)
            self.mass = mass
            self.gravity = gravity
            self.required_variables.append("position_d")
        if self.name == "energy_kinetic" or self.name == "energy_total":
            assert (mass is not None and inertia_matrix is not None)
            self.mass = mass
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["Va", "omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_rotational":
            assert(inertia_matrix is not None)
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_translational":
            assert(mass is not None)
            self.mass = mass
            self.required_variables.append("Va")

    def add_requirement(self, name, variable):
        self.variables[name] = variable

    def calculate_value(self):
        val = 0
        if self.name == "energy_potential" or self.name == "energy_total":
            val += self.mass * self.gravity * (-self.variables["position_d"].value)
        if self.name == "energy_kinetic_rotational" or self.name == "energy_kinetic" or self.name == "energy_total":
            for i, axis in enumerate(["omega_p", "omega_q", "omega_r"]):
                m_i = self.inertia_matrix[i, i]
                val += 1 / 2 * m_i * self.variables[axis].value ** 2
        if self.name == "energy_kinetic_translational" or self.name == "energy_kinetic" or self.name == "energy_total":
            val += 1 / 2 * self.mass * self.variables["Va"].value ** 2

        return val


class Actuation:
    def __init__(self, model_inputs, actuator_inputs, dynamics):
        """
        PyFly actuation object, responsible for verifying validity of configured actuator model, processing inputs and
        actuator dynamics.

        :param model_inputs: ([string]) the states used by PyFly as inputs to dynamics
        :param actuator_inputs: ([string]) the user configured actuator input states
        :param dynamics: ([string]) the user configured actuator states to simulate dynamics for
        """
        self.states = {}
        self.coefficients = [[np.array([]) for _ in range(3)] for __ in range(2)]
        self.elevon_dynamics = False
        self.dynamics = dynamics
        self.inputs = actuator_inputs
        self.model_inputs = model_inputs
        self.input_indices = {s: i for i, s in enumerate(actuator_inputs)}
        self.dynamics_indices = {s: i for i, s in enumerate(dynamics)}

    def set_states(self, values, save=True):
        """
        Set values of actuator states.

        :param values: ([float]) list of state values + list of state derivatives
        :param save: (bool) whether to commit values to state history
        :return:
        """
        for i, state in enumerate(self.dynamics):
            self.states[state].set_value((values[i], values[len(self.dynamics) + i]), save=save)

        # Simulator model operates on elevator and aileron angles, if aircraft has elevon dynamics need to map
        if self.elevon_dynamics:
            elevator, aileron = self._map_elevon_to_elevail(er=self.states["elevon_right"].value,
                                                            el=self.states["elevon_left"].value)

            self.states["aileron"].set_value((aileron, 0), save=save)
            self.states["elevator"].set_value((elevator, 0), save=save)

    def add_state(self, state):
        """
        Add actuator state, and configure dynamics if state has dynamics.

        :param state: (ControlVariable) actuator state
        :return:
        """
        self.states[state.name] = state
        if state.name in self.dynamics:
            for i in range(2):
                for j in range(3):
                    self.coefficients[i][j] = np.append(self.coefficients[i][j], state.coefs[i][j])

    def get_values(self):
        """
        Get state values and derivatives for states in actuator dynamics.

        :return: ([float]) list of state values + list of state derivatives
        """
        return [self.states[state].value for state in self.dynamics] + [self.states[state].dot for state in
                                                                            self.dynamics]

    def rhs(self, setpoints=None):
        """
        Right hand side of actuator differential equation.

        :param setpoints: ([float] or None) setpoints for actuators. If None, setpoints are set as the current command
        of the dynamics variable
        :return: ([float]) right hand side of actuator differential equation.
        """
        if setpoints is None:
            setpoints = [self.states[state].command for state in self.dynamics]
        states = [self.states[state].value for state in self.dynamics]
        dots = [self.states[state].dot for state in self.dynamics]
        dot = np.multiply(states,
                          self.coefficients[0][0]) + np.multiply(setpoints,
                                                                 self.coefficients[0][2]) + np.multiply(dots, self.coefficients[0][1])
        ddot = np.multiply(states,
                           self.coefficients[1][0]) + np.multiply(setpoints,
                                                                  self.coefficients[1][2]) + np.multiply(dots, self.coefficients[1][1])

        return np.concatenate((dot, ddot))

    def set_and_constrain_commands(self, commands):
        """
        Take  raw actuator commands and constrain them according to the state limits and constraints, and update state
        values and history.

        :param commands: ([float]) raw commands
        :return: ([float]) constrained commands
        """
        dynamics_commands = {}
        if self.elevon_dynamics and "elevator" and "aileron" in self.inputs:
            elev_c, ail_c = commands[self.input_indices["elevator"]], commands[self.input_indices["aileron"]]
            elevon_r_c, elevon_l_c = self._map_elevail_to_elevon(elev=elev_c, ail=ail_c)
            dynamics_commands = {"elevon_right": elevon_r_c, "elevon_left": elevon_l_c}

        for state in self.dynamics:
            if state in self.input_indices:
                state_c = commands[self.input_indices[state]]
            else:  # Elevail inputs with elevon dynamics
                state_c = dynamics_commands[state]
            self.states[state].set_command(state_c)
            dynamics_commands[state] = self.states[state].command

        # The elevator and aileron commands constrained by limitatons on physical elevons
        if self.elevon_dynamics:
            elev_c, ail_c = self._map_elevon_to_elevail(er=dynamics_commands["elevon_right"],
                                                        el=dynamics_commands["elevon_left"])
            self.states["elevator"].set_command(elev_c)
            self.states["aileron"].set_command(ail_c)

        for state, i in self.input_indices.items():
            commands[i] = self.states[state].command

        return commands

    def finalize(self):
        """
        Assert valid configuration of actuator dynamics and set actuator state limits if applicable.
        """
        if "elevon_left" in self.dynamics or "elevon_right" in self.dynamics:
            assert("elevon_left" in self.dynamics and "elevon_right" in self.dynamics and not ("aileron" in self.dynamics
                   or "elevator" in self.dynamics))
            assert ("elevon_left" in self.states and "elevon_right" in self.states)
            self.elevon_dynamics = True

            # Set elevator and aileron limits from elevon limits for plotting purposes etc.
            if "elevator" in self.states:
                elev_min, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                          el=self.states["elevon_left"].value_min)
                elev_max, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                          el=self.states["elevon_left"].value_max)
                self.states["elevator"].value_min = elev_min
                self.states["elevator"].value_max = elev_max
            if "aileron" in self.states:
                _, ail_min = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                         el=self.states["elevon_left"].value_min)
                _, ail_max = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                         el=self.states["elevon_left"].value_max)
                self.states["aileron"].value_min = ail_min
                self.states["aileron"].value_max = ail_max

    def reset(self, state_init=None):
        for state in self.dynamics:
            init = None
            if state_init is not None and state in state_init:
                init = state_init[state]
            self.states[state].reset(value=init)

        if self.elevon_dynamics:
            elev, ail = self._map_elevon_to_elevail(er=self.states["elevon_right"].value, el=self.states["elevon_left"].value)
            self.states["elevator"].reset(value=elev)
            self.states["aileron"].reset(value=ail)

    def _map_elevail_to_elevon(self, elev, ail):
        er = -1 * ail + elev
        el = ail + elev
        return er, el

    def _map_elevon_to_elevail(self, er, el):
        ail = (-er + el) / 2
        elev = (er + el) / 2
        return elev, ail


class AttitudeQuaternion:
    def __init__(self):
        """
        Quaternion attitude representation used by PyFly.
        """
        self.quaternion = None
        self.euler_angles = {"roll": None, "pitch": None, "yaw": None}
        self.history = None

    def seed(self, seed):
        return

    def reset(self, euler_init):
        """
        Reset state of attitude quaternion to value given by euler angles.

        :param euler_init: ([float]) the roll, pitch, yaw values to initialize quaternion to.
        """
        if euler_init is not None:
            self._from_euler_angles(euler_init)
        else:
            raise NotImplementedError
        self.history = [self.quaternion]

    def as_euler_angle(self, angle="all", timestep=-1):
        """
        Get attitude quaternion as euler angles, roll, pitch and yaw.

        :param angle: (string) which euler angle to return or all.
        :param timestep: (int) timestep
        :return: (float or dict) requested euler angles.
        """
        e0, e1, e2, e3 = self.history[timestep]
        res = {}
        if angle == "roll" or angle == "all":
            res["roll"] = np.arctan2(2 * (e0 * e1 + e2 * e3), e0 ** 2 + e3 ** 2 - e1 ** 2 - e2 ** 2)
        if angle == "pitch" or angle == "all":
            res["pitch"] = np.arcsin(2 * (e0 * e2 - e1 * e3))
        if angle == "yaw" or angle == "all":
            res["yaw"] = np.arctan2(2 * (e0 * e3 + e1 * e2), e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2)

        return res if angle == "all" else res[angle]

    @property
    def value(self):
        return self.quaternion

    def _from_euler_angles(self, euler):
        """
        Set value of attitude quaternion from euler angles.

        :param euler: ([float]) euler angles roll, pitch, yaw.
        """
        phi, theta, psi = euler
        e0 = np.cos(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)
        e1 = np.cos(psi / 2) * np.cos(theta / 2) * np.sin(phi / 2) - np.sin(psi / 2) * np.sin(theta / 2) * np.cos(
            phi / 2)
        e2 = np.cos(psi / 2) * np.sin(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.cos(theta / 2) * np.sin(
            phi / 2)
        e3 = np.sin(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) - np.cos(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)

        self.quaternion = (e0, e1, e2, e3)

    def set_value(self, quaternion, save=True):
        """
        Set value of attitude quaternion.

        :param quaternion: ([float]) new quaternion value
        :param save: (bool) whether to commit value to history of attitude.
        """
        self.quaternion = quaternion
        if save:
            self.history.append(self.quaternion)


class Wind:
    def __init__(self, turbulence, mag_min=None, mag_max=None, b=None, turbulence_intensity=None, dt=None):
        """
        Wind and turbulence object used by PyFly.

        :param turbulence: (bool) whether turbulence is enabled
        :param mag_min: (float) minimum magnitude of steady wind component
        :param mag_max: (float) maximum magnitude of steady wind component
        :param b: (float) wingspan of aircraft
        :param turbulence_intensity: (string) intensity of turbulence
        :param dt: (float) integration step length
        """
        self.turbulence = turbulence
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.steady = None
        self.components = []
        self.turbulence_sim_length = 250

        if self.turbulence:
            self.dryden = DrydenGustModel(self.turbulence_sim_length, dt, b, intensity=turbulence_intensity)
        else:
            self.dryden = None

        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """
        Seed random number generator of object

        """
        self.np_random = np.random.RandomState(seed)
        if self.turbulence:
            self.dryden.seed(seed)

    def reset(self, value=None):
        """
        Reset wind object to initial state

        :param value: ([float] or float) strength and direction of the n, e and d components or magnitude of the steady wind.
        """
        if value is None or isinstance(value, float) or isinstance(value, int):
            if value is None and self.mag_min is None and self.mag_max is None:
                value = []
                for comp in self.components:
                    comp.reset()
                    value.append(comp.value)
            else:
                if value is None:
                    magnitude = self.np_random.uniform(self.mag_min, self.mag_max)
                else:
                    magnitude = value
                w_n = self.np_random.uniform(-magnitude, magnitude)
                w_e_max = np.sqrt(magnitude ** 2 - w_n ** 2)
                w_e = self.np_random.uniform(-w_e_max, w_e_max)
                w_d = np.sqrt(magnitude ** 2 - w_n ** 2 - w_e ** 2)
                value = [w_n, w_e, w_d]

        if self.turbulence:
            self.dryden.reset()

        self.steady = value
        for i, comp in enumerate(self.components):
            comp.reset(value[i])

    def set_value(self, timestep):
        """
        Set value to wind value at timestep t

        :param timestep: (int) timestep
        """
        value = self.steady

        if self.turbulence:
            value += self._get_turbulence(timestep, "linear")

        for i, comp in enumerate(self.components):
            comp.set_value(value[i])

    def get_turbulence_linear(self, timestep):
        """
        Get linear component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) linear component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "linear")

    def get_turbulence_angular(self, timestep):
        """
        Get angular component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) angular component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "angular")

    def _get_turbulence(self, timestep, component):
        """
        Get turbulence at given timestep.

        :param timestep: (int) timestep
        :param component: (string) which component to return, linear or angular.
        :return: ([float]) turbulence component at timestep
        """
        if timestep >= self.dryden.sim_length:
            self.dryden.simulate(self.turbulence_sim_length)

        if component == "linear":
            return self.dryden.vel_lin[:, timestep]
        else:
            return self.dryden.vel_ang[:, timestep]


class Plot:
    def __init__(self, id, variables=None, title=None, x_unit=None, xlabel=None, ylabel=None, dt=None,
                 plot_quantity=None):
        """
        Plot object used by PyFly to house (sub)figures.

        :param id: (string or int) identifier of figure
        :param variables: ([string]) list of names of states included in figure
        :param title: (string) title of figure
        :param x_unit: (string) unit of x-axis, one of timesteps or seconds
        :param xlabel: (string) label for x-axis
        :param ylabel: (string) label for y-axis
        :param dt: (float) integration step length, required when x_unit is seconds.
        :param plot_quantity: (string) the attribute of the states that is plotted
        """
        self.id = id
        self.title = title

        if x_unit is None:
            x_unit = "timesteps"
        elif x_unit not in ["timesteps", "seconds"]:
            raise Exception("Unsupported x unit (one of timesteps/seconds)")
        elif x_unit == "seconds" and dt is None:
            raise Exception("Parameter dt can not be none when x unit is seconds")

        self.x_unit = x_unit
        self.y_units = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.variables = variables
        self.axis = None
        self.dt = dt
        self.plot_quantity = plot_quantity

    def add_variable(self, var):
        """
        Add state to plot

        :param var: (string) name of state
        """
        if self.variables is None:
            self.variables = []
        self.variables.append(var)

        if var.unit not in self.y_units:
            self.y_units.append(var.unit)

        if len(self.y_units) > 2:
            raise Exception("More than two different units in plot")

    def close(self):
        """
        Close figure
        """
        for var in self.variables:
            var.close_plot(self.id)

        self.axis = None

    def plot(self, fig=None, targets=None):
        """
        Plot history of states in figure

        :param fig: (matplotlib.pyplot.figure) optional parent figure
        :param targets: (dict) target values for states in state name - values pairs
        """
        first = False
        if self.axis is None:
            first = True

            if self.xlabel is not None:
                xlabel = self.xlabel
            else:
                if self.x_unit == "timesteps":
                    xlabel = self.x_unit
                elif self.x_unit == "seconds":
                    xlabel = "Time (s)"

            if self.ylabel is not None:
                ylabel = self.ylabel
            else:
                ylabel = self.y_units[0]

            if fig is not None:
                self.axis = {self.y_units[0]: plt.subplot(fig, title=self.title, xlabel=xlabel, ylabel=ylabel)}
            else:
                self.axis = {self.y_units[0]: plt.plot(title=self.title, xlabel=xlabel, ylabel=ylabel)}

            if len(self.y_units) > 1:
                self.axis[self.y_units[1]] = self.axis[self.y_units[0]].twinx()
                self.axis[self.y_units[1]].set_ylabel(self.y_units[1])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, v in enumerate(self.variables):
            if self.plot_quantity is not None:
                v.plot_quantity = self.plot_quantity
            target = targets[v.name] if targets is not None and v.name in targets else None
            v.plot(self.axis[v.unit], plot_id=self.id, target=target, color=colors[i])

        if first:
            if len(self.y_units) > 1:
                labeled_lines = []
                for ax in self.axis.values():
                    labeled_lines.extend([l for l in ax.lines if "_line" not in l.get_label()])
                self.axis[self.y_units[0]].legend(labeled_lines, [l.get_label() for l in labeled_lines])
            else:
                self.axis[self.y_units[0]].legend()

        else:
            for ax in self.axis.values():
                ax.relim()
                ax.autoscale_view()

        if self.x_unit == "seconds":
            xticks = self.axis[self.y_units[0]].get_xticks()
            self.axis[self.y_units[0]].set_xticklabels(["{0:.1f}".format(tick * self.dt) for tick in xticks])


class PyFly:
    REQUIRED_VARIABLES = ["alpha", "beta", "roll", "pitch", "yaw", "omega_p", "omega_q", "omega_r", "position_n",
                          "position_e", "position_d", "velocity_u", "velocity_v", "velocity_w", "Va",
                          "elevator", "aileron", "rudder", "throttle"]

    def __init__(self,
                 config_path=osp.join(osp.dirname(__file__), "pyfly_config.json"),
                 parameter_path=osp.join(osp.dirname(__file__), "x8_param.mat"),
                 config_kw=None):
        """
        A flight simulator for fixed wing aircraft with configurable initial conditions, constraints and turbulence
        conditions

        :param config_path: (string) Path to json configuration file for simulator
        :param parameter_path: (string) Path to file containing required aircraft parameters
        """

        def set_config_attrs(parent, kws):
            for attr, val in kws.items():
                if isinstance(val, dict):
                    set_config_attrs(parent[attr], val)
                else:
                    parent[attr] = val

        _, parameter_extension = osp.splitext(parameter_path)
        if parameter_extension == ".mat":
            self.params = scipy.io.loadmat(parameter_path, squeeze_me=True)
        elif parameter_extension == ".json":
            with open(parameter_path) as param_file:
                self.params = json.load(param_file)
        else:
            raise Exception("Unsupported parameter file extension.")

        self.I = np.array([[self.params["Jx"], 0, -self.params["Jxz"]],
                           [0, self.params["Jy"], 0, ],
                           [-self.params["Jxz"], 0, self.params["Jz"]]
                           ])

        self.gammas = [self.I[0, 0] * self.I[2, 2] - self.I[0, 2] ** 2]
        self.gammas.append((np.abs(self.I[0, 2]) * (self.I[0, 0] - self.I[1, 1] + self.I[2, 2])) / self.gammas[0])
        self.gammas.append((self.I[2, 2] * (self.I[2, 2] - self.I[1, 1]) + self.I[0, 2] ** 2) / self.gammas[0])
        self.gammas.append(self.I[2, 2] / self.gammas[0])
        self.gammas.append(np.abs(self.I[0, 2]) / self.gammas[0])
        self.gammas.append((self.I[2, 2] - self.I[0, 0]) / self.I[1, 1])
        self.gammas.append(np.abs(self.I[0, 2]) / self.I[1, 1])
        self.gammas.append(((self.I[0, 0] - self.I[1, 1]) * self.I[0, 0] + self.I[0, 2] ** 2) / self.gammas[0])
        self.gammas.append(self.I[0, 0] / self.gammas[0])

        self.params["ar"] = self.params["b"] ** 2 / self.params["S_wing"]  # aspect ratio

        with open(config_path) as config_file:
            self.cfg = json.load(config_file)

        if config_kw is not None:
            set_config_attrs(self.cfg, config_kw)

        self.state = {}
        self.attitude_states = ["roll", "pitch", "yaw"]
        self.actuator_states = ["elevator", "aileron", "rudder", "throttle", "elevon_left", "elevon_right"]
        self.model_inputs = ["elevator", "aileron", "rudder", "throttle"]
        self.energy_states = []

        if not set(self.REQUIRED_VARIABLES).issubset([v["name"] for v in self.cfg["variables"]]):
            raise Exception("Missing required variable(s) in config file: {}".format(
                ",".join(list(set(self.REQUIRED_VARIABLES) - set([v["name"] for v in self.cfg["variables"]])))))

        self.dt = self.cfg["dt"]
        self.rho = self.cfg["rho"]
        self.g = self.cfg["g"]
        self.wind = Wind(mag_min=self.cfg["wind_magnitude_min"], mag_max=self.cfg["wind_magnitude_max"],
                         turbulence=self.cfg["turbulence"], turbulence_intensity=self.cfg["turbulence_intensity"],
                         dt=self.cfg["dt"], b=self.params["b"])

        self.state["attitude"] = AttitudeQuaternion()
        self.attitude_states_with_constraints = []

        self.actuation = Actuation(model_inputs=self.model_inputs,
                                   actuator_inputs=self.cfg["actuation"]["inputs"],
                                   dynamics=self.cfg["actuation"]["dynamics"])

        for v in self.cfg["variables"]:
            if v["name"] in self.attitude_states and any([v.get(attribute, None) is not None for attribute in
                                                          ["constraint_min", "constraint_max", "value_min",
                                                           "value_max"]]):
                self.attitude_states_with_constraints.append(v["name"])
            if v["name"] in self.actuator_states:
                self.state[v["name"]] = ControlVariable(**v)
                self.actuation.add_state(self.state[v["name"]])
            elif "energy" in v["name"]:
                self.energy_states.append(v["name"])
                self.state[v["name"]] = EnergyVariable(mass=self.params["mass"], inertia_matrix=self.I, gravity=self.g, **v)
            else:
                self.state[v["name"]] = Variable(**v)

            if "wind" in v["name"]:
                self.wind.components.append(self.state[v["name"]])

        for state in self.model_inputs:
            if state not in self.state:
                self.state[state] = ControlVariable(name=state, disabled=True)
                self.actuation.add_state(self.state[state])

        self.actuation.finalize()

        for energy_state in self.energy_states:
            for req_var_name in self.state[energy_state].required_variables:
                self.state[energy_state].add_requirement(req_var_name, self.state[req_var_name])

        # TODO: check that all plotted variables are declared in cfg.variables
        self.plots = []
        for i, p in enumerate(self.cfg["plots"]):
            vars = p.pop("states")
            p["dt"] = self.dt
            p["id"] = i
            self.plots.append(Plot(**p))

            for v_name in vars:
                self.plots[-1].add_variable(self.state[v_name])

        self.cur_sim_step = None
        self.viewer = None

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator

        :param seed: (int) seed for random state
        """
        for i, var in enumerate(self.state.values()):
            var.seed(seed + i)

        self.wind.seed(seed)

    def reset(self, state=None):
        """
        Reset state of simulator. Must be called before first use.

        :param state: (dict) set initial value of states to given value.
        """
        self.cur_sim_step = 0

        for name, var in self.state.items():
            if name in ["Va", "alpha", "beta", "attitude"] or "wind" in name or "energy" in name or isinstance(var, ControlVariable):
                continue
            var_init = state[name] if state is not None and name in state else None
            var.reset(value=var_init)

        self.actuation.reset(state)

        wind_init = None
        if state is not None:
            if "wind" in state:
                wind_init = state["wind"]
            elif all([comp in state for comp in ["wind_n", "wind_e", "wind_d"]]):
                wind_init = [state["wind_n"], state["wind_e"], state["wind_d"]]
        self.wind.reset(wind_init)

        Theta = self.get_states_vector(["roll", "pitch", "yaw"])
        vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))

        Va, alpha, beta = self._calculate_airspeed_factors(Theta, vel)
        self.state["Va"].reset(Va)
        self.state["alpha"].reset(alpha)
        self.state["beta"].reset(beta)

        self.state["attitude"].reset(Theta)

        for energy_state in self.energy_states:
            self.state[energy_state].reset(self.state[energy_state].calculate_value())

    def render(self, mode="plot", close=False, viewer=None, targets=None, block=False):
        """
        Visualize history of simulator states.

        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param close: (bool) close figure after showing
        :param viewer: (dict) viewer object with figure and gridspec that pyfly will attach plots to
        :param targets: (dict) string list pairs of states with target values added to plots containing these states.
        """
        if mode == "plot":
            if viewer is not None:
                self.viewer = viewer
            elif self.viewer is None:
                self.viewer = {"fig": plt.figure(figsize=(9, 16))}
                subfig_count = len(self.plots)
                self.viewer["gs"] = matplotlib.gridspec.GridSpec(subfig_count, 1)

            for i, p in enumerate(self.plots):
                sub_fig = self.viewer["gs"][i, 0] if p.axis is None else None
                if targets is not None:
                    plot_variables = [v.name for v in p.variables]
                    plot_targets = list(set(targets).intersection(plot_variables))
                    p.plot(fig=sub_fig, targets={k: v for k, v in targets.items() if k in plot_targets})
                else:
                    p.plot(fig=sub_fig)

            if viewer is None:
                plt.show(block=block)
            if close:
                for p in self.plots:
                    p.close()
                self.viewer = None
        elif mode == "animation":
            raise NotImplementedError
        else:
            raise ValueError("Unexpected value {} for mode".format(mode))

    def step(self, commands):
        """
        Perform one integration step from t to t + dt.

        :param commands: ([float]) actuator setpoints
        :return: (bool, dict) whether integration step was successfully performed, reason for step failure
        """
        success = True
        info = {}

        # Record command history and apply conditions on actuator setpoints
        control_inputs = self.actuation.set_and_constrain_commands(commands)

        y0 = list(self.state["attitude"].value)
        y0.extend(self.get_states_vector(["omega_p", "omega_q", "omega_r", "position_n", "position_e", "position_d",
                                          "velocity_u", "velocity_v", "velocity_w"]))
        y0.extend(self.actuation.get_values())
        y0 = np.array(y0)

        try:
            sol = scipy.integrate.solve_ivp(fun=lambda t, y: self._dynamics(t, y), t_span=(0, self.dt),
                                            y0=y0)
            self._set_states_from_ode_solution(sol.y[:, -1], save=True)

            Theta = self.get_states_vector(["roll", "pitch", "yaw"])
            vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))

            Va, alpha, beta = self._calculate_airspeed_factors(Theta, vel)
            self.state["Va"].set_value(Va)
            self.state["alpha"].set_value(alpha)
            self.state["beta"].set_value(beta)

            for energy_state in self.energy_states:
                self.state[energy_state].set_value(self.state[energy_state].calculate_value(), save=True)

            self.wind.set_value(self.cur_sim_step)
        except ConstraintException as e:
            success = False
            info = {"termination": e.variable}

        self.cur_sim_step += 1

        return success, info

    def get_states_vector(self, states, attribute="value"):
        """
        Get attribute of multiple states.

        :param states: ([string]) list of state names
        :param attribute: (string) state attribute to retrieve
        :return: ([?]) list of attribute for each state
        """
        return [getattr(self.state[state_name], attribute) for state_name in states]

    def save_history(self, path, states):
        """
        Save simulator state history to file.

        :param path: (string) path to save history to
        :param states: (string or [string]) names of states to save
        """
        res = {}
        if states == "all":
            save_states = self.state.keys()
        else:
            save_states = states

        for state in save_states:
            res[state] = self.state[state].history

        np.save(path, res)

    def _dynamics(self, t, y, control_sp=None):
        """
        Right hand side of dynamics differential equation.

        :param t: (float) current integration time
        :param y: ([float]) current values of integration states
        :param control_sp: ([float]) setpoints for actuators
        :return: ([float]) right hand side of differential equations
        """

        if t > 0:
            self._set_states_from_ode_solution(y, save=False)

        attitude = y[:4]

        omega = self.get_states_vector(["omega_p", "omega_q", "omega_r"])
        vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))
        u_states = self.get_states_vector(self.model_inputs)

        f, tau = self._forces(attitude, omega, vel, u_states)

        return np.concatenate([
            self._f_attitude_dot(t, attitude, omega),
            self._f_omega_dot(t, omega, tau),
            self._f_p_dot(t, vel, attitude),
            self._f_v_dot(t, vel, omega, f),
            self._f_u_dot(t, control_sp)
        ])

    def _forces(self, attitude, omega, vel, controls):
        """
        Get aerodynamic forces acting on aircraft.

        :param attitude: ([float]) attitude quaternion of aircraft
        :param omega: ([float]) angular velocity of aircraft
        :param vel: ([float]) linear velocity of aircraft
        :param controls: ([float]) state of actutators
        :return: ([float], [float]) forces and moments in x, y, z of aircraft frame
        """
        elevator, aileron, rudder, throttle = controls

        p, q, r = omega

        if self.wind.turbulence:
            p_w, q_w, r_w = self.wind.get_turbulence_angular(self.cur_sim_step)
            p, q, r = p - p_w, q - q_w, r - r_w

        Va, alpha, beta = self._calculate_airspeed_factors(attitude, vel)

        Va = self.state["Va"].apply_conditions(Va)
        alpha = self.state["alpha"].apply_conditions(alpha)
        beta = self.state["beta"].apply_conditions(beta)

        pre_fac = 0.5 * self.rho * Va ** 2 * self.params["S_wing"]

        e0, e1, e2, e3 = attitude
        fg_b = self.params["mass"] * self.g * np.array([2 * (e1 * e3 - e2 * e0),
                                                        2 * (e2 * e3 + e1 * e0),
                                                        e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2])

        C_L_alpha_lin = self.params["C_L_0"] + self.params["C_L_alpha"] * alpha

        # Nonlinear version of lift coefficient with stall
        a_0 = self.params["a_0"]
        M = self.params["M"]
        e = self.params["e"]  # oswald efficiency
        ar = self.params["ar"]
        C_D_p = self.params["C_D_p"]
        C_m_fp = self.params["C_m_fp"]
        C_m_alpha = self.params["C_m_alpha"]
        C_m_0 = self.params["C_m_0"]

        sigma = (1 + np.exp(-M * (alpha - a_0)) + np.exp(M * (alpha + a_0))) / (
                    (1 + np.exp(-M * (alpha - a_0))) * (1 + np.exp(M * (alpha + a_0))))
        C_L_alpha = (1 - sigma) * C_L_alpha_lin + sigma * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))

        f_lift_s = pre_fac * (C_L_alpha + self.params["C_L_q"] * self.params["c"] / (2 * Va) * q + self.params[
            "C_L_delta_e"] * elevator)
        # C_D_alpha = self.params["C_D_0"] + self.params["C_D_alpha1"] * alpha + self.params["C_D_alpha2"] * alpha ** 2
        C_D_alpha = C_D_p + (1 - sigma) * (self.params["C_L_0"] + self.params["C_L_alpha"] * alpha) ** 2 / (
                    np.pi * e * ar) + sigma * (2 * np.sign(alpha) * math.pow(np.sin(alpha), 3))
        C_D_beta = self.params["C_D_beta1"] * beta + self.params["C_D_beta2"] * beta ** 2
        f_drag_s = pre_fac * (
                    C_D_alpha + C_D_beta + self.params["C_D_q"] * self.params["c"] / (2 * Va) * q + self.params[
                "C_D_delta_e"] * elevator ** 2)

        C_m = (1 - sigma) * (C_m_0 + C_m_alpha * alpha) + sigma * (C_m_fp * np.sign(alpha) * np.sin(alpha) ** 2)
        m = pre_fac * self.params["c"] * (C_m + self.params["C_m_q"] * self.params["b"] / (2 * Va) * q + self.params[
            "C_m_delta_e"] * elevator)

        f_y = pre_fac * (
                    self.params["C_Y_0"] + self.params["C_Y_beta"] * beta + self.params["C_Y_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_Y_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_Y_delta_a"] * aileron + self.params["C_Y_delta_r"] * rudder)
        l = pre_fac * self.params["b"] * (
                    self.params["C_l_0"] + self.params["C_l_beta"] * beta + self.params["C_l_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_l_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_l_delta_a"] * aileron + self.params["C_l_delta_r"] * rudder)
        n = pre_fac * self.params["b"] * (
                    self.params["C_n_0"] + self.params["C_n_beta"] * beta + self.params["C_n_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_n_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_n_delta_a"] * aileron + self.params["C_n_delta_r"] * rudder)

        f_aero = np.dot(self._rot_b_v(np.array([0, alpha, beta])), np.array([-f_drag_s, f_y, -f_lift_s]))
        tau_aero = np.array([l, m, n])

        Vd = Va + throttle * (self.params["k_motor"] - Va)
        f_prop = np.array([0.5 * self.rho * self.params["S_prop"] * self.params["C_prop"] * Vd * (Vd - Va), 0, 0])
        tau_prop = np.array([-self.params["k_T_P"] * (self.params["k_Omega"] * throttle) ** 2, 0, 0])

        f = f_prop + fg_b + f_aero
        tau = tau_aero + tau_prop

        return f, tau

    def _f_attitude_dot(self, t, attitude, omega):
        """
        Right hand side of quaternion attitude differential equation.

        :param t: (float) time of integration
        :param attitude: ([float]) attitude quaternion
        :param omega: ([float]) angular velocity
        :return: ([float]) right hand side of quaternion attitude differential equation.
        """
        p, q, r = omega
        T = np.array([[0, -p, -q, -r],
                      [p, 0, r, -q],
                      [q, -r, 0, p],
                      [r, q, -p, 0]
                      ])
        return 0.5 * np.dot(T, attitude)

    def _f_omega_dot(self, t, omega, tau):
        """
        Right hand side of angular velocity differential equation.

        :param t: (float) time of integration
        :param omega: ([float]) angular velocity
        :param tau: ([float]) moments acting on aircraft
        :return: ([float]) right hand side of angular velocity differential equation.
        """
        return np.array([
            self.gammas[1] * omega[0] * omega[1] - self.gammas[2] * omega[1] * omega[2] + self.gammas[3] * tau[0] +
            self.gammas[4] * tau[2],
            self.gammas[5] * omega[0] * omega[2] - self.gammas[6] * (omega[0] ** 2 - omega[2] ** 2) + tau[1] / self.I[
                1, 1],
            self.gammas[7] * omega[0] * omega[1] - self.gammas[1] * omega[1] * omega[2] + self.gammas[4] * tau[0] +
            self.gammas[8] * tau[2]
        ])

    def _f_v_dot(self, t, v, omega, f):
        """
        Right hand side of linear velocity differential equation.

        :param t: (float) time of integration
        :param v: ([float]) linear velocity
        :param omega: ([float]) angular velocity
        :param f: ([float]) forces acting on aircraft
        :return: ([float]) right hand side of linear velocity differntial equation.
        """
        v_dot = np.array([
            omega[2] * v[1] - omega[1] * v[2] + f[0] / self.params["mass"],
            omega[0] * v[2] - omega[2] * v[0] + f[1] / self.params["mass"],
            omega[1] * v[0] - omega[0] * v[1] + f[2] / self.params["mass"]
        ])

        return v_dot

    def _f_p_dot(self, t, v, attitude):
        """
        Right hand side of position differential equation.

        :param t: (float) time of integration
        :param v: ([float]) linear velocity
        :param attitude: ([float]) attitude quaternion
        :return: ([float]) right hand side of position differntial equation.
        """
        e0, e1, e2, e3 = attitude
        T = np.array([[e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
                      [2 * (e1 * e2 + e3 * e0), e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2, 2 * (e2 * e3 - e1 * e0)],
                      [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2]
                      ])
        return np.dot(T, v)

    def _f_u_dot(self, t, setpoints):
        """
        Right hand side of actuator differential equation.

        :param t: (float) time of integration
        :param setpoints: ([float]) setpoint for actuators
        :return: ([float]) right hand side of actuator differential equation.
        """
        return self.actuation.rhs(setpoints)

    def _rot_b_v(self, attitude):
        """
        Rotate vector from body frame to vehicle frame.

        :param Theta: ([float]) vector to rotate, either as Euler angles or quaternion
        :return: ([float]) rotated vector
        """
        if len(attitude) == 3:
            phi, th, psi = attitude
            return np.array([
                [np.cos(th) * np.cos(psi), np.cos(th) * np.sin(psi), -np.sin(th)],
                [np.sin(phi) * np.sin(th) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                 np.sin(phi) * np.sin(th) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.sin(phi) * np.cos(th)],
                [np.cos(phi) * np.sin(th) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                 np.cos(phi) * np.sin(th) * np.sin(psi) - np.sin(phi) * np.cos(psi), np.cos(phi) * np.cos(th)]
            ])
        elif len(attitude) == 4:
            e0, e1, e2, e3 = attitude
            return np.array([[-1 + 2 * (e0 ** 2 + e1 ** 2), 2 * (e1 * e2 + e3 * e0), 2 * (e1 * e3 - e2 * e0)],
                             [2 * (e1 * e2 - e3 * e0), -1 + 2 * (e0 ** 2 + e2 ** 2), 2 * (e2 * e3 + e1 * e0)],
                             [2 * (e1 * e3 + e2 * e0), 2 * (e2 * e3 - e1 * e0), -1 + 2 * (e0 ** 2 + e3 ** 2)]])

        else:
            raise ValueError("Attitude is neither Euler angles nor Quaternion")

    def _rot_v_b(self, Theta):
        """
        Rotate vector from vehicle frame to body frame.

        :param Theta: ([float]) vector to rotate
        :return: ([float]) rotated vector
        """
        phi, th, psi = Theta
        return np.array([
            [np.cos(th) * np.cos(psi), np.sin(phi) * np.sin(th) * np.cos(psi) - np.cos(phi) * np.sin(psi),
             np.cos(phi) * np.sin(th) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
            [np.cos(th) * np.sin(psi), np.sin(phi) * np.sin(th) * np.sin(psi) + np.cos(phi) * np.cos(psi),
             np.cos(phi) * np.sin(th) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
            [-np.sin(th), np.sin(phi) * np.cos(th), np.cos(phi) * np.cos(th)]
        ])

    def _calculate_airspeed_factors(self, attitude, vel):
        """
        Calculate the airspeed factors airspeed (Va), angle of attack (alpha) and sideslip angle (beta).

        :param attitude: ([float]) attitude quaternion
        :param vel: ([float]) linear velocity
        :return: ([float]) airspeed factors Va, alpha, beta
        """
        if self.wind.turbulence:
            turbulence = self.wind.get_turbulence_linear(self.cur_sim_step)
        else:
            turbulence = np.zeros(3)

        wind_vec = np.dot(self._rot_b_v(attitude), self.wind.steady) + turbulence
        airspeed_vec = vel - wind_vec

        Va = np.linalg.norm(airspeed_vec)
        alpha = np.arctan2(airspeed_vec[2], airspeed_vec[0])
        beta = np.arcsin(airspeed_vec[1] / Va)

        return Va, alpha, beta

    def _set_states_from_ode_solution(self, ode_sol, save):
        """
        Set states from ODE solution vector.

        :param ode_sol: ([float]) solution vector from ODE solver
        :param save: (bool) whether to save values to state history, i.e. whether solution represents final step
        solution or intermediate values during integration.
        """
        self.state["attitude"].set_value(ode_sol[:4] / np.linalg.norm(ode_sol[:4]))
        if save:
            euler_angles = self.state["attitude"].as_euler_angle()
            self.state["roll"].set_value(euler_angles["roll"], save=save)
            self.state["pitch"].set_value(euler_angles["pitch"], save=save)
            self.state["yaw"].set_value(euler_angles["yaw"], save=save)
        else:
            for state in self.attitude_states_with_constraints:
                self.state[state].set_value(self.state["attitude"].as_euler_angle(state))
        start_i = 4

        self.state["omega_p"].set_value(ode_sol[start_i], save=save)
        self.state["omega_q"].set_value(ode_sol[start_i + 1], save=save)
        self.state["omega_r"].set_value(ode_sol[start_i + 2], save=save)
        self.state["position_n"].set_value(ode_sol[start_i + 3], save=save)
        self.state["position_e"].set_value(ode_sol[start_i + 4], save=save)
        self.state["position_d"].set_value(ode_sol[start_i + 5], save=save)
        self.state["velocity_u"].set_value(ode_sol[start_i + 6], save=save)
        self.state["velocity_v"].set_value(ode_sol[start_i + 7], save=save)
        self.state["velocity_w"].set_value(ode_sol[start_i + 8], save=save)
        self.actuation.set_states(ode_sol[start_i + 9:], save=save)


if __name__ == "__main__":
    from dryden import DrydenGustModel
    from pid_controller import PIDController

    pfly = PyFly("pyfly_config.json", "x8_param.mat")
    pfly.seed(0)

    pid = PIDController(pfly.dt)
    pid.set_reference(phi=0.2, theta=0, va=22)

    pfly.reset(state={"roll": -0.5, "pitch": 0.15})

    for i in range(500):
        phi = pfly.state["roll"].value
        theta = pfly.state["pitch"].value
        Va = pfly.state["Va"].value
        omega = [pfly.state["omega_p"].value, pfly.state["omega_q"].value, pfly.state["omega_r"].value]

        action = pid.get_action(phi, theta, Va, omega)
        success, step_info = pfly.step(action)

        if not success:
            break

    pfly.render(block=True)
else:
    from .dryden import DrydenGustModel
