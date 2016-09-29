# name this file solutions.py
"""Volume 2 Lab 14: Optimization Packages II (CVXOPT)
Derek Miller
Shane Company
13 Jan 2016
"""

import numpy as np
import scipy as sp
from cvxopt import matrix
from cvxopt import solvers

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + y + 3z     >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    # minimize this function
    c = matrix([2.,1.,3.])
    # given these constraints
    G = matrix([[-1.,-2.,-1.,0.,0.],[-2.,-1.,0.,-1.,0.],[0.,-3.,0.,0.,-1.]])
    # constrained by these values
    h = matrix([-3.,-10.,0.,0.,0.])
    # compute the solution to the optimization problem
    sol = solvers.lp(c,G,h)
    return sol['x'], sol['primal objective']



def prob2():
    """Solve the transportation problem by converting all equality constraints
    into inequality constraints.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    # minimize this function
    c = matrix([4.,7.,6.,8.,8.,9.])
    # constraints
    G = matrix([[1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,-1.,0.,0.,0.,0.,0.],
                [1.,-1.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,-1.,0.,0.,0.,0.],
                [0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,-1.,0.,0.,0.],
                [0.,0.,1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,0.,-1.,0.,0.],
                [0.,0.,0.,0.,1.,-1.,1.,-1.,0.,0.,0.,0.,0.,0.,-1.,0.],
                [0.,0.,0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,-1.]])
    # with respect to these values
    h = matrix([7.,-7.,2.,-2.,4.,-4.,5.,-5.,8.,-8.,0.,0.,0.,0.,0.,0.])
    # compute the solution to the modified constraints
    sol = solvers.lp(c,G,h)
    return sol['x'], sol['primal objective']


def prob3():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    Q = matrix([[3., 2. ,1.],[2., 4., 2.],[1., 2., 3.]])
    p = matrix([3., 0., 1.])
    sol = solvers.qp(Q,p)
    return sol['x'], sol['primal objective']


def prob4():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective']*-1000)
    """
    data = np.load('ForestData.npy')
    A = np.array([[1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                  [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                  [0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                  [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                  [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.],
                  [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.],
                  [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.]])
    C1 = -1*data[:,4]
    C2 = -1*data[:,5]
    C3 = -1*data[:,6]
    I = -1*np.eye(21)
    h = matrix([75.,90.,140.,60.,212.,98.,113.,-75.,-90.,-140.,-60.,-212.,-98.,-113.,-40000.,-5.,-55160.,
                0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    G = np.vstack((A,-1*A,C1,C2,C3,I))
    print(G.shape)
    G = matrix(G)
    cT = matrix(-1*data[:,3])
    sol = solvers.lp(cT,G,h)
    return sol['x'],-1000*sol['primal objective']

# Problem 4 generates a 38x21 matrix



###TEST FUNCTION###

def t(x):
    if x == 1:
        ans = prob1()
        print(ans[0])
        print(ans[1])
        # ANS: 10
    elif x == 2:
        ans = prob2()
        print(ans[0])
        print(ans[1])
        # ANS: 86
    elif x == 3:
        one,two = prob3()
        print(one)
        print(two)
        # ANS: -2.5
    elif x == 4:
        one,two = prob4()
        print(one)
        print(two)
        # ANS: 322,515,000
    else:
        print("Invalid Input")

