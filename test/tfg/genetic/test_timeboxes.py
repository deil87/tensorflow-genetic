import unittest

import numpy as np
from sympy import *
from sympy import Symbol

from tfg.genetic.Evolution import Evolution


class TimeboxesTestCase(unittest.TestCase):

    def test_solve_equation_iteratively(self):
        x = 0
        total = 100
        n_boxes = 5
        idxs = list(range(1, n_boxes + 1))
        def timeboxes(input):
            return list(map(lambda idx: np.exp(input * idx), idxs))

        while True:
            x = x + 0.000001
            print("X = " + str(x))
            delta = np.abs(sum(timeboxes(x)) - 100)
            if delta < 1e-3:
                break
            if delta > 100:
                "No solution was found"
                break

        print("X = " + str(x)) # X = 2.25261000000629
        # for 5 : X = 0.8063150000023487

    def test_timeboxes_approximate_exponential(self):
        N = 2
        A = 10000 / np.exp(N)

        def L(x):
            return A * np.exp(x)

        def R(x) :
            return L(x) - L(x - 1)

        segments = list(map(R, range(1, N+ 1)))
        res = list([A]) + segments
        print(res)
        sum_res = sum(res)
        print(sum_res)


    def test_timeboxes_exponential_exact(self):
        max_runtime = 10000
        segments = Evolution(None, None, 1234).generate_timeboxes(3, max_runtime)

        sum_of_segments = sum(segments)
        self.assertEqual(max_runtime, sum_of_segments)

    # def test_timeboxes(self):
    #     n_boxes = 5
    #     max_runtime = 100
    #     idxs = list(range(1, n_boxes + 1))
    #     factorial = math.factorial(n_boxes - 1)
    #     log_of_mrt = np.log(max_runtime)
    #     tmp_term = (log_of_mrt / float(factorial))
    #     time_unit = tmp_term ** (1 / float(n_boxes))
    #
    #     time_unit = 0.8063150000023487
    #     timebox_exponents =  list(map(lambda x: x * time_unit, idxs))
    #     timeboxes = list(map(lambda x: np.exp(x), timebox_exponents))
    #     sum_of_tbs =  sum(timeboxes)
    #     print("Time unit" + str(time_unit))


    def test_timeboxes(self):
        from sympy.parsing.sympy_parser import parse_expr
        x = Symbol("x", real=True)
        sympy_exp = parse_expr('exp(x) + exp(2*x) -100')
        # sympy_exp = parse_expr('exp(x) + exp(2*x)+ exp(3*x)+ exp(4*x)+ exp(5*x) -100')
        # res3 = sympy_exp.evalf(5, subs={x: 3})
        # print(res3)
        solved = solve(sympy_exp)
        print(solved)
        evaluated = solved[0].evalf()
        print(evaluated)
        print(solved[1].evalf())


if __name__ == '__main__':
    unittest.main()
