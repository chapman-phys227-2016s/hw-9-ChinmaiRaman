#! /usr/bin/env python

"""
File: Backward2.py
Copyright (c) 2016 Chinmai Raman
License: MIT
Course: PHYS227
Assignment: 9.11
Date: May 5th, 2016
Email: raman105@mail.chapman.edu
Name: Chinmai Raman
Description: Compares the accuracy of a one-sided three-point, second-order differentiating formula with a two-point, second-order differentiating formula.
"""
from __future__ import division
import numpy as np
from unittest import TestCase

class Diff:
    def __init__(self, f, h = 1e-5):
        self.f = f
        self.h = float(h)

class Backward1(Diff):
    def __init__(self, f, h = 1e-5):
        Diff.__init__(self, f, h)

    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x) - f(x - h)) / h

class Backward2(Diff):
    def __init__(self, f, h = 1e-5):
        Diff.__init__(self, f, h)

    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x - 2 * h) - 4 * f(x - h) + 3 * f(x)) / (2 * h)

def methods():
    lst = []
    for name in globals().keys():
        if name[0].isupper():
            if issubclass(eval(name), Diff):
                if name != 'Diff':
                    lst.append(name)
    return lst

def table(f, x, h_values, dfdx = None):
    arr = methods()
    print '     h       ',
    for name in arr:
        print '%-15s' % name,
        print '%-15s' % (name + ':error'),
    print
    for h in h_values:
        print '%10.2E' % h,
        for name in arr:
            if dfdx is not None:
                d = eval(name)(f, h)
                out = d(x)
                out2 = d(x) - dfdx
            print '%15.8E' % out,
            print '%17.8E' % out2,
        print

def main():
    g = lambda t: np.exp(-t)

    dgdt = lambda t: -1 * np.exp(-t)
    var = dgdt(0)

    return table(g, 0, [2**(-k) for k in xrange(15)], var)

if __name__ == '__main__':
    main()

class Test_Diff(TestCase):
    def test_diff(self):
        g = lambda t: np.exp(-t)
        dgdt = lambda t: -1 * np.exp(-t)
        der_actual = dgdt(0)
        arr = methods()
        tol = 1e-5
        for method in arr:
            inst = eval(method)(g)
            assert(abs(inst(0) - der_actual) < tol), 'Failure'