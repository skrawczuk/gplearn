from typing import Callable
import sys

sys.path.append('gplearn')
import numpy as np
import sympy as sp
import pandas as pd
import time
import mpmath
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness


def taylor(f, a, n):
    if isinstance(f, Callable):
        coefs = mpmath.taylor(f, a, n)
    else:
        coefs = mpmath.taylor(sp.lambdify(x, f), a, n)
    coefs = np.array([float(i) for i in coefs])
    return coefs


def f_combined(x, base):
    return eval('{b}.exp(x) + {b}.sin(x)'.format(b=base))


def f_exp(x, base):
    return eval('{b}.exp(2*x)'.format(b=base))


def f_sin(x, base):
    return eval('{b}.sin(2*x)'.format(b=base))


def coefs_fit(f):
    f_np = lambda x: f(x, 'np')
    a_fit = np.polyfit(x_data, y, deg)[::-1]
    return a_fit


if __name__ == '__main__':

    # deg = 10,13, 15
    # f = sin, exp, combined
    # parsimony= 0.001, 0.005, 0.01, 0.05,
    # 3 its each

    x, X0, k = sp.symbols('x,X0,k')
    x_data = np.linspace(-1, 1, 1000)

    results = pd.DataFrame(columns=['degree', 'function', 'loss function', 'loss value', 'sparsity', 'timing'])

    for deg in [10, 13, 15]:
        for f in [f_sin, f_exp, f_combined]:
            for p_c in [0.001, 0.005, 0.01, 0.05]:
                for it in range(3):
                    y = f(x_data, 'np')
                    a_n = coefs_fit(f)
                    n = np.arange(len(a_n), dtype=int).reshape(-1, 1)

                    sr_w = SymbolicRegressor(population_size=25000,
                                             generations=15,
                                             stopping_criteria=1e-7,
                                             p_crossover=0.7,
                                             p_subtree_mutation=0.1,
                                             p_hoist_mutation=0.05,
                                             p_point_mutation=0.1,
                                             const_range=(-1, 1),
                                             max_samples=1.0,
                                             verbose=True,
                                             parsimony_coefficient=p_c,  # 0.01
                                             random_state=None,  # 5
                                             function_set=['add', 'mul', 'sub', 'div', 'factorial', 'power', 'alt_sin',
                                                           'alt_cos'],
                                             metric='weighted_mae'
                                             )
                    t1 = time.time()
                    sr_w.fit(n, a_n)
                    t2_w = time.time() - t1

                    sr = SymbolicRegressor(population_size=25000,
                                           generations=15,
                                           stopping_criteria=1e-7,
                                           p_crossover=0.7,
                                           p_subtree_mutation=0.1,
                                           p_hoist_mutation=0.05,
                                           p_point_mutation=0.1,
                                           const_range=(-1, 1),
                                           max_samples=1.0,
                                           verbose=True,
                                           parsimony_coefficient=p_c,  # 0.01
                                           random_state=None,  # 5
                                           function_set=['add', 'mul', 'sub', 'div', 'factorial', 'power', 'alt_sin',
                                                         'alt_cos'],
                                           )
                    t1 = time.time()
                    sr.fit(n, a_n)
                    t2 = time.time() - t1

                    acc_w = np.linalg.norm(sr_w.predict(n) - a_n)
                    acc = np.linalg.norm(sr.predict(n) - a_n)

                    results.loc[len(results)] = [deg, str(f(x, 'sp')), 'mae', acc, p_c, t2]
                    results.loc[len(results)] = [deg, str(f(x, 'sp')), 'w_mae', acc_w, p_c, t2_w]

    results.to_csv('results')
