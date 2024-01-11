#!/usr/bin/env python
import logging
import unittest
from appoppy.control_loop import AbstractActuator, AbstractSensor, IntegralController, PIDController
import numpy as np
import astropy.units as u


class A1DSensor(AbstractSensor):

    def __init__(self):
        self.measure = 42.
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
        self.measure = np.array([12, 23, 34, 45.])

    def get_measurement(self):
        if self._linked_to is None:
            return self.measure
        else:
            actuator_value = self._linked_to.get_command()
            return self._function(actuator_value)

    @property
    def dimension(self):
        return len(self.measure)

    def _identity_function(self, x):
        return x

    def link_to(self, actuator, function=None):
        self._linked_to = actuator
        if function is None:
            self._function = self._identity_function
        else:
            self._function = function


class AMultiDActuator(AbstractActuator):

    def __init__(self):
        self._cmd = np.zeros(2)

    def set_command(self, command):
        self._cmd = command

    def get_command(self):
        return self._cmd

    @property
    def dimension(self):
        return len(self._cmd)


class IntegralControllerTest(unittest.TestCase):

    def setUp(self):
        self._setUpBasicLogging()

    def _setUpBasicLogging(self):
        logging.basicConfig(level=logging.DEBUG)


    def test_1d(self):
        actuator = A1DActuator()
        sensor = A1DSensor()
        sensor.link_to(actuator)
        controller = IntegralController(sensor, actuator, name='1D')
        controller.set_point = 40
        controller.ki = 0.5
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 20)
        np.testing.assert_allclose(sensor.get_measurement(), 20)
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 30)
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), 35)

    def test_multi_dimension(self):

        def _actuator2d_to_sensor4d(x):
            '''
            act=[x,y] -> sens=[x,y,x,y]
            '''
            return np.tile(x, 2)

        def _control_matrix_4_in_2_average():
            return 0.5*np.tile(np.eye(2), (2, 1)).T

        actuator = AMultiDActuator()
        sensor = AMultiDSensor()
        sensor.link_to(actuator, function=_actuator2d_to_sensor4d)
        controller = IntegralController(sensor, actuator, name='MultiD')
        controller.control_matrix = _control_matrix_4_in_2_average()
        controller.ki = 0.5
        controller.set_point = np.array([1, 10, 3, 30])
        # should eventually converge to [2, 20]
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), [1, 10])
        controller.step()
        np.testing.assert_allclose(actuator.get_command(), [1.5, 15])

    def test_integrator_converge(self):

        def _bad_calibration(x):
            return 1.2*x

        actuator = A1DActuator()
        actuator.set_command(0)
        sensor = A1DSensor()
        sensor.link_to(actuator, function=_bad_calibration)
        controller = IntegralController(
            sensor, actuator, name='Integrator converges')
        controller.set_point = 3.14
        controller.ki = 1.0
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
        controller = IntegralController(
            sensor, actuator, name='Bad Calibration')
        controller.set_point = 3.14
        controller.ki = 0.5
        for i in range(30):
            controller.step()
        self.assertGreater(np.abs(sensor.get_measurement()), 1e6)

    def test_integrator_high_gain_diverge(self):

        def _identity(x):
            return x

        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        sensor.link_to(actuator, function=_identity)
        controller = IntegralController(sensor, actuator, name='High Gain')
        controller.set_point = 3.14
        controller.ki = 2.5
        for i in range(30):
            controller.step()
        self.assertGreater(np.abs(sensor.get_measurement()), 1e6)

    def test_initialize_command_to_zero(self):
        actuator = A1DActuator()
        actuator.set_command(40)
        sensor = A1DSensor()
        controller = IntegralController(sensor, actuator)
        np.testing.assert_almost_equal(controller.last_command, 0)

    def test_delay(self):
        actuator = A1DActuator()
        actuator.set_command(0)
        sensor = A1DSensor()
        sensor.link_to(actuator)
        controller = IntegralController(
            sensor, actuator, delay=2, name='Delay')
        controller.set_point = 42
        controller.ki = 1.0
        controller.step()
        self.assertAlmostEqual(controller.last_command, 0)
        self.assertAlmostEqual(sensor.get_measurement(), 0)
        controller.step()
        self.assertAlmostEqual(controller.last_command, 0)
        self.assertAlmostEqual(sensor.get_measurement(), 0)
        controller.step()
        self.assertAlmostEqual(controller.last_command, 42)
        self.assertAlmostEqual(sensor.get_measurement(), 42)

    # def test_sense_and_correct(self):

    #     def _identity(x):
    #         return x

    #     actuator = A1DActuator()
    #     actuator.set_command(40)
    #     sensor = A1DSensor()
    #     sensor.link_to(actuator, function=_identity)
    #     controller = IntegralController(sensor, actuator)
    #     controller.set_point = 3.14
    #     controller.ki = 0.5
    #     meas = sensor.get_measurement()
    #     cmd = controller.compute_command(meas)
    #     self.assertAlmostEqual(cmd, 40-0.5*(40-3.14))


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
