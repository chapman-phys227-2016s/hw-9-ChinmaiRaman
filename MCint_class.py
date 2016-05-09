#! /usr/bin/env python

"""
File: MCint_class.py
Copyright (c) 2016 Chinmai Raman
License: MIT
Course: PHYS227
Assignment: 9.14
Date: May 8th, 2016
Email: raman105@mail.chapman.edu
Name: Chinmai Raman
Description: Implements Monte Carlo integration.
"""
from __future__ import division
import numpy as np
from unittest import TestCase

class Integrator(object):
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n

    def construct_method(self):
        raise NotImplementedError('no rule in class %s' %
                                  self.__class__.__name__)

    def integrate(self, f):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i]*f(self.points[i])
        return s

    def vectorized_integrate(self, f):
        return np.dot(self.weights, f(self.points))

class MCint_vec(Integrator):
    def construct_method_og(self, f, points_per_batch = 1000000):
        s = 0
        a, b, n = self.a, self.b, self.n
        x = np.random.uniform(a, b, n)
        w = np.zeros(len(x)) + (float(b - a) / n * s)
        return x, w

    def construct_method(self, f, points_per_batch = 1000000):
        s = 0
        a, b, n = self.a, self.b, self.n
        rest = n % points_per_batch
        batch_sizes = [points_per_batch] * (n // points_per_batch) + [rest]
        for elem in batch_sizes:
            x = np.random.uniform(a, b, elem)
            s += np.sum(f(x))
        I = (float(b - a) / n) * s
        return I

class Test_MCint_vec(TestCase):
    def test_MCint_vec(self):
        def f(x): return 2 * x
        montecarlo = MCint_vec(0, 10, 35750000)
        msg = 'The MCint_vec class fails to integrate correctly. The correct definite integral is 100.00'
        tol = 1e-2
        assert(abs(montecarlo.construct_method(f, 10000000) - 100.000000000) < tol), msg