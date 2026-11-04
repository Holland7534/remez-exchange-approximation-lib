"""Tests for the Remez exchange approximation library."""

import math
import unittest

from remez_exchange_approximation_lib import RemezResult, remez


class RemezBasicTests(unittest.TestCase):
    def test_constant_function_is_exact(self):
        result = remez(lambda x: 3.0, -1.0, 1.0, 0)
        self.assertIsInstance(result, RemezResult)
        self.assertEqual(len(result.coefficients), 1)
        self.assertAlmostEqual(result.coefficients[0], 3.0, places=10)
        self.assertLess(result.max_error, 1e-10)

    def test_linear_function_is_exact(self):
        result = remez(lambda x: 2.0 * x + 1.0, -1.0, 1.0, 1)
        self.assertEqual(len(result.coefficients), 2)
        self.assertAlmostEqual(result.coefficients[0], 1.0, places=8)
        self.assertAlmostEqual(result.coefficients[1], 2.0, places=8)
        self.assertLess(result.max_error, 1e-8)

    def test_quadratic_function_is_exact(self):
        result = remez(lambda x: x * x, -1.0, 1.0, 2)
        self.assertEqual(len(result.coefficients), 3)
        self.assertAlmostEqual(result.coefficients[0], 0.0, places=6)
        self.assertAlmostEqual(result.coefficients[1], 0.0, places=6)
        self.assertAlmostEqual(result.coefficients[2], 1.0, places=6)
        self.assertLess(result.max_error, 1e-6)

    def test_sine_approximation_error_decreases_with_degree(self):
        low = remez(math.sin, -1.0, 1.0, 3)
        high = remez(math.sin, -1.0, 1.0, 5)
        self.assertLess(high.max_error, low.max_error)
        self.assertLess(high.max_error, 1e-4)

    def test_result_nodes_are_sorted_and_within_interval(self):
        result = remez(math.exp, -2.0, 2.0, 4)
        nodes = list(result.nodes)
        self.assertEqual(nodes, sorted(nodes))
        for x in nodes:
            self.assertGreaterEqual(x, -2.0)
            self.assertLessEqual(x, 2.0)

    def test_iterations_are_positive(self):
        result = remez(math.sin, -1.0, 1.0, 4)
        self.assertGreaterEqual(result.iterations, 1)

    def test_converged_flag_is_boolean(self):
        result = remez(math.cos, -1.0, 1.0, 4)
        self.assertIsInstance(result.converged, bool)


class RemezEdgeCaseTests(unittest.TestCase):
    def test_invalid_interval_raises(self):
        with self.assertRaises(ValueError):
            remez(math.sin, 1.0, 1.0, 2)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            remez(math.sin, -1.0, 1.0, -1)

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            remez(math.sin, -1.0, 1.0, 2, samples=1)

    def test_zero_max_iterations_raises(self):
        with self.assertRaises(ValueError):
            remez(math.sin, -1.0, 1.0, 2, max_iterations=0)

    def test_max_iterations_caps_iterations(self):
        result = remez(math.sin, -1.0, 1.0, 6, max_iterations=1)
        self.assertEqual(result.iterations, 1)

    def test_tight_tolerance_still_returns_result(self):
        result = remez(math.sin, -1.0, 1.0, 4, tolerance=0.0)
        self.assertIsInstance(result, RemezResult)
        self.assertEqual(len(result.coefficients), 5)

    def test_very_small_interval(self):
        result = remez(lambda x: x * x, 1.0, 1.0 + 1e-6, 2)
        self.assertLess(result.max_error, 1e-6)

    def test_high_degree_polynomial_reproduces_itself(self):
        # f is itself a degree-5 polynomial; a degree-5 fit should be exact.
        def f(x):
            return 1.0 + x - 0.5 * x ** 2 + 0.25 * x ** 3 - 0.125 * x ** 4 + 0.0625 * x ** 5

        result = remez(f, -1.0, 1.0, 5)
        self.assertLess(result.max_error, 1e-6)
        self.assertAlmostEqual(result.coefficients[0], 1.0, places=5)
        self.assertAlmostEqual(result.coefficients[1], 1.0, places=5)
        self.assertAlmostEqual(result.coefficients[2], -0.5, places=5)
        self.assertAlmostEqual(result.coefficients[3], 0.25, places=5)
        self.assertAlmostEqual(result.coefficients[4], -0.125, places=5)
        self.assertAlmostEqual(result.coefficients[5], 0.0625, places=5)


if __name__ == "__main__":
    unittest.main()
