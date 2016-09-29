# name this file 'solutions.py'.
"""Volume II Lab 16: Simplex
Derek Miller
Vol 2
28 Jan

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""
from __future__ import division
import numpy as np
from scipy import linalg as la

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    
    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        for entry in b:
            if entry < 0:
                raise ValueError("system is infeasible at the origin")
        self.c = c
        self.A = A
        self.b = b
        self.basic = len(b) # number of basic variables
        self.nonbasic = len(c) # number of nonbasic variables
        L = range(len(b) + len(c)) # an ordered list of indices
        # switch the indices of the basic and nonbasic variables
        # nonbasic variables, basic variables = basic variables, nonbasic variables
        # basic first and nonbasic second
        self.vars = L
        self.T = None

    def gen_tableau(self):
        """
        Generates the initialized tableau matrix as follows:
    
        T = [0, -c^T, 1]
            [b, A_bar,0]
    
        where   b = self.b,
                c = self.c,
                A_bar = self.A stacked with an Identity matrix of size self.basic

        returns T
        """
        # the width of tableau T
        w_T = self.A.shape[0] + self.A.shape[1] + 2
        # the length of tableau T
        l_T = len(self.b) + 1
        # initialize tableau as an array of zeros
        T = np.zeros((l_T, w_T))
        # change the last entry of first row to a 1
        T[0,-1] = 1
        # add -c to the first row of the tableau T
        cbar = np.hstack((self.c,np.zeros_like(self.b)))
        T[0,1:-1] = -1*cbar
        # add b to the first column of T (skipping the first row)
        T[1:,0] = self.b
        # create A_bar = A + identity matrix
        A_bar = np.hstack((self.A,np.eye(self.basic)))
        # add A_bar to the correct place in the matrix
        T[1:,1:-1] = A_bar
        return T

    def find_pivot(self, T):
        """
        Determines pivot row and pivot column
        Input:  T (nparray) the tableau
        Output: p (tuple) pivot row and pivot column
        """
        r = 0
        c = 0
        # iterate through the first row of the tableau
        for t1 in range(1, len(T[0])):
            # find the first negative value and assign it to r (pivot row)
            if T[0,t1] < 0:
                c = t1
                #print(T[:,c])
                for t2 in range(1,len(T[:,c])):
                    if (T[1:,c] <= 0).all():
                        raise ValueError("Houston has an unbounded problem")
                    else:
                        ratios = np.absolute(T[1:,0]/T[1:,c])
                        r = np.argmin(ratios)+1
                break
        print("PIVOT: row %d, column %d") % (r,c)
        if c == 0:
            return "done"
        else:
            return (r,c)

    def pivot(self,T,row_col):
        """
        Checks for unboundedness and performs one pivot to completion.
        """
        # get the pivot row and column
        if row_col == "done":
            return T
        else:
            # take row r and divide by T[r,c]
            r,c = row_col
            T[r,:] /= T[r,c]
            for index in range(len(T[:,c])):
                if index == r:
                    continue
                else:
                    T[index,:] -= T[r,:] * T[index,c]
        return T

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        self.T = self.gen_tableau()
        print("\n\n\n>>>BEGIN<<<")
        print(self.basic)
        print(self.nonbasic)
        print(self.T)
        row_col = "not done"
        while row_col != "done":
            row_col = self.find_pivot(self.T)
            T = self.pivot(self.T,row_col)
            self.T = T
            #x = raw_input()
            print("\n---New Tableau---")
            print(self.T)
        max_val = self.T[0,0]
        basic_vars = dict()
        nonbasic_vars = dict()
        for i in range(len(T[0,1:])):
            # if the objective equation term is zero
            if T[0,i+1] == 0:
                # add this to the basic variables
                basic = np.argwhere(T[:,i+1] != 0)[0,0]
                basic_vars[i] = T[basic,0]
            else:
                # add to nonbasic variables
                nonbasic_vars[i] = 0
        return max_val, basic_vars, nonbasic_vars


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    data = np.load(filename)
    A = data['A']
    p = data['p']
    m = data['m']
    d = data['d']
    print A #3x4 resource coefficients
    print p #1x4 maximize
    print m #1x3 resource constraint
    print d #1x4 demand constraint
    A_I = np.vstack((A,np.eye(4))) #7x4 coefficients
    m_d = np.hstack((m,d)) #1x7 constraints
    poop = raw_input()
    brains = SimplexSolver(1./p,A_I,m_d)
    one,two,three = brains.solve()
    print one
    print two
    print three


def test():
    c = np.array([3,2])
    A = np.array([[1,-1],[3,1],[4,3]])
    b = np.array([2,5,7])
    brain = SimplexSolver(c,A,b)
    maxi,paxi,taxi = brain.solve()
    print(maxi)
    print(paxi[0])
    print(paxi[1])

# END OF FILE =================================================================
