
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
    '''

    def __init__(self, sensor, actuator):
        self._sensor = sensor
        self._actuator = actuator
        self._last_command = self._actuator.get_command()
        self.ki = 0.5
        self.set_point = 0
        self.control_matrix = np.eye(actuator.dimension, sensor.dimension)

    def step(self):
        meas = self._sensor.get_measurement()
        error = meas - self.set_point
        delta_command = np.dot(self.control_matrix, error)
        command = self._last_command - self.ki * delta_command
        self._actuator.set_command(command)
        self._last_command = self._actuator.get_command()


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
