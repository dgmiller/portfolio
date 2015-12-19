# spec.py

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

def centered_difference_quotient(f,pts,h = 1e-5):
    '''
    Compute the centered difference quotient for function (f)
    given points (pts).
    Inputs:
        f (function): the function for which the derivative will be approximated
        pts (array): array of values to calculate the derivative
    Returns:
        centered difference quotient (array): array of the centered difference
            quotient
    '''
    CDQ = [] # centered difference quotient list
    Df_app = lambda x: .5*(f(x+h)-f(x-h))/h # function to compute CDQ
    for x in pts:
        CDQ.append(Df_app(x))
    return np.array(CDQ)


def jacobian(f,n,m,pt,h = 1e-5):
    '''
    Compute the approximate Jacobian matrix of f at pt using the centered
    difference quotient.
    Inputs:
        f (function): the multidimensional function for which the derivative
            will be approximated
        n (int): dimension of the domain of f
        m (int): dimension of the range of f
        pt (array): an n-dimensional array representing a point in R^n
        h (float): a float to use in the centered difference approximation
    Returns:
        Jacobian matrix of f at pt using the centered difference quotient.
    '''
    e = np.eye(n,m) # the identity vectors from R^n to R^m
    J = np.zeros((m,n)) # the jacobian matrix
    for i in range(n): # we only need to worry about columns
        Df = lambda x: .5*(f(x+h*e[i]) - f(x-h*e[i]))/h
        # the func Df() will take in a row vector with n columns
        # we multiply h by the ith row of e
        # 'convert' the result to a column when added to J
        J[:,i] = Df(pt)
    return J


def findError():
    '''
    Compute the maximum error of your jacobian function for the function
    f(x,y)=[(e^x)*sin(y)+y^3,3y-cos(x)] on the square [-1,1]x[-1,1].
    Returns:
        Maximum error of your jacobian function.
    '''
    err = []
    X = np.linspace(-1,1,100)
    Y = np.linspace(-1,1,100)
    f = lambda x: np.array([np.exp(x[0]) * np.sin(x[1]) + x[1]**3, 3*x[1]-np.cos(x[0])])
    for i in X:
        for j in Y:
            J = jacobian(f,2,2,np.array([i,j]))
            true_J = np.array([
                [np.exp(i)*np.sin(j), np.exp(i)*np.cos(j)+3*j**2],
                [np.sin(i), 3]])
            err.append(la.norm(true_J-J))
    return max(err)
        

def Filter(image,F):
    '''
    Applies the filter to the image.
    Inputs:
        image (array): an array of the image
        F (array): an nxn filter to be applied (a numpy array).
    Returns:
        The filtered image.
    '''
    m,n = image.shape
    h,k = F.shape
    image_pad = np.zeros((m+2*h,n+2*k))
    image_pad[h:h+m, k:k+n] = image
    C = np.zeros(image.shape)
    for i in range(m):
        for j in range(n):
            C[i,j] = np.trace(F.T * image_pad[i:i+h, j:j+k])
    return C


def sobelFilter(image):
    '''
    Applies the Sobel filter to the image
    Inputs:
        image(array): an array of the image in grayscale
    Returns:
        The image with the Sobel filter applied.
    '''
    A = image
    S = np.array([
        [-1./8, 0, 1./8],
        [-2./8, 0, 2./8],
        [-1./8, 0, 1./8]])
    X = Filter(A, S)
    Y = Filter(A, S.T)
    B = np.sqrt(X**2 + Y**2) # This B is the gradient of A
    m = B.max()
    B /= m
    M = B.mean()*4
    B[B>M] = 1.
    B[B<=M] = 0.
    return B

        