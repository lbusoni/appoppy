#!/usr/bin/env python
import unittest
from appoppy.control_loop import AbstractActuator, AbstractSensor, IntegralController, PIDController
import numpy as np
import astropy.units as u


class A1DSensor(AbstractSensor):

    def __init__(self):
        self.measure = 42
        self._linked_to = None

    def get_measurement(self):
        if self._linked_to is None:
            return self.measure
        else:
            actuator_value = self._linked_to.get_command()
            return self._function(actuator_value)

    @property
    def dimension(self):
        return 1

    def _identity_function(self, x):
        return x

    def link_to(self, actuator, function=None):
        self._linked_to = actuator
        if function is None:
            self._function = self._identity_function
        else:
            self._function = function


class A1DActuator(AbstractActuator):

    def __init__(self):
        self._cmd = 0

    def set_command(self, command):
        self._cmd = command

    def get_command(self):
        return self._cmd

    @property
    def dimension(self):
        return 1


class AMultiDSensor(AbstractSensor):

    def __init__(self):
        self.measure = np.array([12, 23, 34])

    def get_measurement(self):
        return self.measure

    @property
    def dimension(self):
        return len(self.measure)


class AMultiDActuator(AbstractActuator):

    def __init__(self):
        self._cmd = np.zeros(4)

    def set_command(self, command):
        self._cmd = command

    def get_command(self):
        return self._cmd

    @property
    def dimension(self):
        return len(self._cmd)


class ControllerTest(unittest.TestCase):

    def test_1d(self):
        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        sensor.link_to(actuator)
        controller = IntegralController(sensor, actuator)
        controller.ki = 0.5
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 20)
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 10)
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 5)

    def test_multi_dimension(self):
        actuator = AMultiDActuator()
        actuator.set_command(np.array([0, 1, 2, 3]))
        sensor = AMultiDSensor()
        sensor.measure = np.ones(3)
        controller = IntegralController(sensor, actuator)
        # identity with last row empty
        controller.control_matrix = np.eye(4, 3)
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), [-0.5, 0.5, 1.5, 3])
        sensor.measure = actuator.get_command()[0:3]
        controller.step()
        np.testing.assert_allclose(
            actuator.get_command(), [-0.25, 0.25, 0.75, 3])

    def test_integrator_converge(self):

        def _bad_calibration(x):
            return 1.2*x

        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        sensor.link_to(actuator, function=_bad_calibration)
        controller = IntegralController(sensor, actuator)
        controller.set_point = 3.14
        controller.ki = 0.5
        for i in range(100):
            controller.step()
        np.testing.assert_allclose(sensor.get_measurement(), 3.14)

    def test_integrator_badly_calibrated_diverge(self):

        def _very_bad_calibration(x):
            return 5*x

        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        sensor.link_to(actuator, function=_very_bad_calibration)
        controller = IntegralController(sensor, actuator)
        controller.set_point = 3.14
        controller.ki = 0.5
        for i in range(100):
            controller.step()
        self.assertGreater(sensor.get_measurement(), 1e6)

    def test_integrator_high_gain_diverge(self):

        def _identity(x):
            return x

        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        sensor.link_to(actuator, function=_identity)
        controller = IntegralController(sensor, actuator)
        controller.set_point = 3.14
        controller.ki = 2.5
        for i in range(100):
            controller.step()
        self.assertGreater(sensor.get_measurement(), 1e6)


class PIDControllerTest(unittest.TestCase):

    def test_1d(self):
        actuator = A1DActuator()
        actuator.set_command(0)
        sensor = A1DSensor()
        sensor.measure = 40-actuator.get_command()
        controller = PIDController(sensor, actuator)
        controller.kp = 0.5
        controller.ki = 0.
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), -20)
        sensor.measure = 40-actuator.get_command()
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), -30)
        sensor.measure = 40-actuator.get_command()
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), -35)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
