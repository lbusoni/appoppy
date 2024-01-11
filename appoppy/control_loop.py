
import logging
import numpy as np
from abc import ABC, abstractmethod


class AbstractActuator(ABC):

    @abstractmethod
    def set_command(self):
        pass

    @abstractmethod
    def get_command(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass


class AbstractSensor(ABC):

    @abstractmethod
    def get_measurement(self):
        pass

    @property
    @abstractmethod
    def dimension(self):
        pass


class IntegralController():
    '''
    Implement an integral controller
    
    Sequence: 
    1) sense, compute command & add it to delay line
    2) actuate
    
    '''

    def __init__(self, sensor, actuator, ki=0.5, delay=0, name='Integral Controller'):
        self._logger = logging.getLogger(name)
        self._sensor = sensor
        self._actuator = actuator
        self._cnt = 0
        self.delay = delay
        self._init_command_history_to_zero()
        self.ki = ki
        self.set_point = 0
        self.control_matrix = np.eye(actuator.dimension, sensor.dimension)

    def _init_command_history(self):
        curr_command = self._get_actuator_cmd()
        self._cmd_hist = np.tile(curr_command, (self.delay+1, 1))
        self._logger.debug(
            "Initialized command history to %s" % self._cmd_hist)

    def _init_command_history_to_zero(self):
        curr_command = self._get_actuator_cmd()*0
        self._cmd_hist = np.tile(curr_command, (self.delay+1, 1))
        self._logger.debug(
            "Initialized command history to %s" % self._cmd_hist)

    def _get_actuator_cmd(self):
        return np.atleast_1d(self._actuator.get_command()).astype(float)

    def _get_sensor_measurement(self):
        return np.atleast_1d(self._sensor.get_measurement()).astype(float)

    def _update_command_history(self, new_value):
        newv = self._cmd_hist.copy()
        newv[0] = new_value
        newv = np.roll(newv, 1, axis=0)
        self._cmd_hist = newv

    @property
    def last_command(self):
        '''
        Last command applied
        '''
        return self._cmd_hist[0]

    @property
    def last_measurement(self):
        '''
        Last measurement done by the sensor
        '''
        return self._last_meas

    def step(self):
        self.sense()
        self.actuate()

    def sense(self):
        # TODO: force sequence sense-actuate-sense-actuate etc
        self._last_meas = self._get_sensor_measurement()
        self._error = self._last_meas - self.set_point
        self._delta_command = -self.ki * \
            np.dot(self.control_matrix, self._error)
        self._total_command = self.last_command + self._delta_command
        self._logger.debug("step %d - meas %s - delta_cmd %s - total_cmd %s" % (
            self._cnt, self.last_measurement, self._delta_command,
            self._total_command))

    def actuate(self):
        self._update_command_history(self._total_command)
        self._actuator.set_command(self.last_command)
        self._logger.debug("step %d - Cmd %s applied" %
                           (self._cnt, self.last_command))
        self._cnt += 1


class PIDController:
    def __init__(self, sensor, actuator, control_matrix=None, kp=1.0, ki=0.0, kd=0.0):
        """
        Initialize the PID controller.

        :param sensor: An instance of Sensor providing measurements.
        :param actuator: An instance of Actuator executing commands.
        :param control_matrix: Control matrix converting from Sensor space to Actuator space.
                              Must be a 2D numpy array with shape (M, N).
        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain.
        """
        self.sensor = sensor
        self.actuator = actuator

        self.set_point = 0

        # Check if control matrix is provided and has valid dimensions
        if control_matrix is not None and len(control_matrix.shape) == 2:
            self.control_matrix = control_matrix
        else:
            # Default to an identity matrix with dimensions (M, N)
            self.control_matrix = np.eye(actuator.dimension, sensor.dimension)

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = np.zeros(sensor.dimension)
        self.integral = np.zeros(sensor.dimension)

    def step(self):
        """
        Compute the control output using PID algorithm.

        :return: Control output.
        """
        # Get current measurement
        measurement = self.sensor.get_measurement()

        # Calculate error
        error = self.set_point - measurement

        # Update integral term
        self.integral += error

        # Calculate PID terms
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error)

        # Calculate control output in sensor space
        control_output_sensor_space = p_term + i_term + d_term

        # Transform control output to Actuator space
        control_output_actuator_space = np.dot(
            self.control_matrix, control_output_sensor_space)

        # Execute the control output using the Actuator
        self.actuator.set_command(control_output_actuator_space)

        # Save current error for the next iteration
        self.prev_error = error

        return control_output_actuator_space
