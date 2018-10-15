# Merging Derek's Optimal Reentry into Original

import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import erf
import matplotlib.pyplot as plt

# Ones Derek doesn't have
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from scipy.optimize import root

### PROBLEM 1 ###

## constants ##
S = 26600. # technically S/2m = 53200
rho_0 = 2.704e-3
beta = 4.26
R = 209.
g = 3.2172e-4
T = 230

## helper functions ##
rho = lambda xi: rho_0 * np.exp(-beta*R*xi)
CD = lambda u: 1.174 - .9*np.cos(u)
CL = lambda u: .6*np.sin(u)

## BVP equations ##

def vdot(v,gamma,xi,u):
    term1 = -S*rho(xi)*v**2*CD(u)
    term2 = -g*np.sin(gamma)/(1+xi)**2
    return term1 + term2

def gammadot(v,gamma,xi,u):
    term1 = S*rho(xi)*v*CL(u)
    term2 = v*np.cos(gamma)/(R*(1+xi))
    term3 = - g*np.cos(gamma)/(v*(1+xi)**2)
    return term1 + term2 + term3
    
def xidot(v,gamma,xi,u):
    return v*np.sin(gamma)/R

def H(v,gamma,xi,Lv,Lgamma,Lxi):
    u =  np.arctan((6*Lgamma)/(9*v*Lv )) 
    Rho = rho_0*np.exp(-beta*R*xi)

    term1 = 10*v**3*np.sqrt(Rho)
    term2 = -S*Rho*v**2*CD(u) - g*np.sin(gamma)/(1+xi)**2
    term3 = S*Rho*v*CL(u) + v*np.cos(gamma)/(R*(1 + xi)) - g*np.cos(gamma)/(v*(1+xi)**2)
    term4 = v*np.sin(gamma)/R
    return term1 + Lv*term2 + Lgamma*term3 + Lxi*term4

def Lvdot(v,gamma,xi,Lv,Lgamma,Lxi,Rho,u):
    term1 = 30*np.sqrt(Rho)*v**2
    term2 = -2*S*Rho*v*CD(u)
    term3 = S*Rho*CL(u) + np.cos(gamma)/(R*(1+xi)) + g*np.cos(gamma)/((1+xi)**2*v**2)
    term4 = np.sin(gamma)/R
    dHdv = term1 + Lv*term2 + Lgamma*term3 + Lxi*term4
    return -dHdv
           
def Lgammadot(v,gamma,xi,Lv,Lgamma,Lxi,Rho,u):
    term1 = -(g/(1+xi)**2)*np.cos(gamma)
    term2 = -v*np.sin(gamma)/(R*(1+xi)) + g*np.sin(gamma)/(v*(1+xi)**2)
    term3 = v*np.cos(gamma)/R
    dHdgamma = Lv*term1 + Lgamma*term2 + Lxi*term3
    return -dHdgamma
    
def  Lxidot(v,gamma,xi,Lv,Lgamma,Lxi,Rho,u):
    term0 = 5*v**3.*np.sqrt(Rho)*(-beta*R)
    term1 = S*beta*R*Rho*v**2*CD(u) + 2*g*np.sin(gamma)/(1+xi)**3
    term2 = -S*beta*R*Rho*v*CL(u) + -v*np.cos(gamma)/(R*(1+xi)**2) + 2*g*np.cos(gamma)/(v*(1+xi)**3)
    dHdxi = term0 + Lv*term1 + Lgamma*term2
    return -dHdxi

def  Tdot(F, u, Rho):
    "The derivative ofTimewith respect to time."
    return 0

### SOLVE THE BVP ###
def ode(x,y):
    """
    Parameters:
        x: independent variable (unused in our ODEs)
        F: vector-valued dependent variable, equal to 
           (v, gam, xi, p1, p2, p3, T); it is an ndnp.array with shape (7,)
    Returns:
        ndnp.array of length (7,) that evalutes the RHS of the ODES
    """
    v, gamma, xi, p1, p2, p3,Time= y
    u =  np.arctan((6*p2)/(9*v*p1 ))
    Rho = rho_0*np.exp(-beta*R*xi)
    out = Time*np.array([ vdot(v,gamma,xi,u),  
                          gammadot(v,gamma,xi,u),
                          xidot(v,gamma,xi,u), 
                          Lvdot(v, gamma, xi, p1, p2, p3, Rho, u),
                          Lgammadot(v,gamma,xi,p1,p2,p3,Rho,u), 
                          Lxidot(v,gamma,xi,p1,p2,p3,Rho,u), 
                          Tdot(y, u, Rho)])
    return out

def bcs(ya,yb):
    """
    Boundary conditions for the BVP
    Parameters:
        ya: data at moment of entry
        yb: data at end of maneuver
    Returns:
        out1: array of initial conditions
        out2: array of endpoint conditions
    """
    v0,gamma0,xi0,Lv0,Lgamma0,Lxi0,Time0 = ya
    vT,gammaT,xiT,LvT,LgammaT,LxiT,TimeT = yb
    out1 = np.array([ v0-.36,
                      gamma0+8.1*np.pi/180,
                      xi0-4/R])
    out2 = np.array([ vT-.27,
                      gammaT,
                      xiT-2.5/R,
                      H(vT,gammaT,xiT,LvT,LgammaT,LxiT)])
    return out1, out2

### AUXILIARY BVP ###

p1, p2, p3 = 1.3, 4.5, .5
def guess_auxiliary(x):
	out = np.array([ .5*(.36+.27)-.5*(.36-.27)*np.tanh(.025*(x-.45*T)),
		np.pi/180.*(.5*(-8.1 + 0)-.5*(-8.1 - 0)*np.tanh(.025*(x-.25*T)) ),
		(1./R)*( .5*(4+2.5)-.5*(4-2.5)*np.tanh(.03*(x-.3*T)) -
		1.4*np.cosh(.025*(x-.25*T))**(-2.) ),
		p1*np.ones(x.shape),
		p2*np.ones(x.shape),
		p3*np.ones(x.shape)])
	return out

def ode_auxiliary(x,y):
    v,gamma,xi,Lv,Lgamma,Lxi = y
    u_aux = Lv*erf( Lgamma*(Lxi-(1.*x)/T) )
    Rho = rho_0*np.exp(-beta*R*xi)
    out = np.array([ vdot(v,gamma,xi,u_aux),
                     gammadot(v,gamma,xi,u_aux),
                     xidot(v,gamma,xi,u_aux),
                     0,
                     0,
                     0 ])
    return out

def bcs_auxiliary(ya,yb):
    v0,gamma0,xi0,Lv0,Lgamma0,Lxi0 = ya
    vT,gammaT,xiT,LvT,LgammaT,LxiT = yb
    out1 = np.array([ v0-.36,
                      gamma0+8.1*np.pi/180,
                      xi0-4/R])
    out2 = np.array([ vT-.27,
                      gammaT,
                      xiT-2.5/R])
    return out1, out2


### SOLVE WITH SCIKITS ###
################################################################
#--------------------------------------------------------------#
def solve_using_scikits(T=230,N=240):
    from scikits import bvp_solver
    problem_auxiliary = bvp_solver.ProblemDefinition( num_ODE = 6,
                                                      num_parameters = 0,
                                                      num_left_boundary_conditions = 3,
                                                      boundary_points = (0, T),
                                                      function = ode_auxiliary,
                                                      boundary_conditions = bcs_auxiliary )
    
    solution_auxiliary = bvp_solver.solve( problem_auxiliary,
                                           solution_guess = guess_auxiliary,
                                           trace = 0,
                                           max_subintervals = 20000 )
    
    x_guess = np.linspace(0,T,N+1) # 240 time steps to the guess of 230 as the final time
    # the solution to the auxiliary BVP gives a good initial guess for the original BVP
    yyy = guess_auxiliary(x_guess)
    initial_guess = solution_auxiliary(x_guess)
        # redefine T to be 230?
    T = x_guess[-1]
    # Generate a guess for the solution of the ODE. These values are specified in the text.
    # v,gamma,xi,Lv,Lgamma,Lxi = sol_guess
    # add a Time vector as shown below
    sol_guess = np.concatenate( (initial_guess,T*np.ones((1,len(x_guess)))) ,axis=0)
    # Lv
    sol_guess[3,:] = -1 # Lv < 0 since cos(u) > 0
    p1, p2, p3 = sol_guess[3,0], sol_guess[4,0], sol_guess[5,0] # initial values to iterate through
    # approximate optimal control u
    u = p1*erf( p2*(p3-x_guess/T) )
    aux_u = u.copy()
    # Lgamma
    # derived from tan(u) = 6*Lgamma/(9*v*Lv) => Lgamma = (9/6)*v*Lv*tan(u)
    sol_guess[4,:] = 1.5*sol_guess[0,:]*sol_guess[3,:]*np.tan(u)
    # Lxi
    # derived from Hamiltonian
    L = list()
    L2 = list()
    G = list()
    for j in range(len(sol_guess[5,:])):
        y = sol_guess[:6,j] # use y to prevent memory handling errors
        # you can verify that crazy things happen by uncommenting the L, L2, and G lists
        # and uncomment their corresponding plot functions below
        #L.append(y[1])
        def new_func(x):
            #y[1] = y[1]
            #L2.append(y[1])
            #gamma = y[1]
            #G.append(gamma)
            if y[1] < 0 and y[1] > -.05: 
                #print "1\t",j
                y[1] = -.05
            if y[1] > 0 and y[1] < .05: 
                #print "2\t",j
                y[1] = .05
            y[5] = x
            return H(y[0],y[1],y[2],y[3],y[4],y[5]) # use y to prevent memory handling errors
        sol = root(new_func,-8)
        if j>0:
            if sol.success == True: 
                sol_guess[5,j] = sol.x
            else: 
                sol_guess[5,j] = sol_guess[5,j-1]
        else: 
            if sol.success == True: 
                sol_guess[5,0] = sol.x
    #plt.plot(L)
    #plt.plot(L2)
    #plt.title("y[1] (blue) and y[1]=y[1] (red)")
    #plt.show()
    #plt.plot(G)
    #plt.title("gamma, same as y[1]=y[1] but no longer converges")
    #plt.show()
    problem = bvp_solver.ProblemDefinition( num_ODE = 7,
                                            num_parameters = 0,
                                            num_left_boundary_conditions = 3,
                                            boundary_points = (0., 1),
                                            function = ode, 
                                            boundary_conditions = bcs )
                                        
    solution = bvp_solver.solve( problem,
                                 solution_guess = sol_guess,
                                 initial_mesh = np.linspace(0,1,len(x_guess)),
                                 max_subintervals=1000,
                                 trace = 1 )
    
    # For more info on the available options for bvp_solver, look at 
    # the docstrings for bvp_solver.ProblemDefinition and bvp_solver.solve
        
    numerical_soln = solution(np.linspace(0,1,N+1))
    u =  np.arctan((6*numerical_soln[4,:])/(9*numerical_soln[0,:]*numerical_soln[3,:] )) 
    domain = np.linspace(0,numerical_soln[6,0],N+1)

    ### PLOT SOLUTIONS ###
    names = ["Velocity","Angle","Altitude","p1","p2","p3"]
    fig = plt.figure(figsize=(20,20))
    for i in xrange(6):
        if i<3:
            ax = fig.add_subplot(2,3,i+1)
            ax.plot(yyy[i,:],':',color='gray',lw=8,alpha=.3,label="guess")
            ax.plot(initial_guess[i,:],'--',lw=4,color='gray',alpha=.5,label="auxiliary")
            ax.plot(numerical_soln[i,:],color='r',lw=2,label="optimal")
            ax.set(title=names[i])
            if i == 0:
                ax.legend(loc='best')
        else:
            ax = fig.add_subplot(2,3,i+1)
            ax.plot(numerical_soln[i,:],color='gray',alpha=.5,lw=8)
            ax.set_title("BVP Costate: %s" % names[i])
    fig.suptitle("Auxiliary vs Optimal BVP")
    plt.show()
    
    soln =  ( domain,
              numerical_soln[0,:],
              numerical_soln[1,:],
              numerical_soln[2,:], 
              numerical_soln[3,:],
              numerical_soln[4,:],
              numerical_soln[5,:],
              u )
    
    
    
    
    
    ################################################################
    
    def plot_reentry_trajectory():
        plt.figure(figsize=(10,10))
        plt.plot(u,color='red',lw=2)
        plt.plot(aux_u[::-1],'--',color='gray',lw=8,alpha=.5)
        plt.title("optimal control u")
        plt.show()
   
    
    def plot_graph_from_book(var):
        if 1:
            # plt.rc("font", size=16)
            host = host_subplot(111, axes_class=AA.Axes)
            plt.subplots_adjust(right=0.75)
    
            par1 = host.twinx()
            par2 = host.twinx()
    
            offset = 80
            new_fixed_axis = par2.get_grid_helper().new_fixed_axis
            par2.axis["right"] = new_fixed_axis(loc='right',axes=par2,offset=(offset, 0))
    
            par2.axis["right"].toggle(all=True)
            host.set_xlim(0, var[0][-1])
            host.set_ylim(.26, .38)
    
            host.set_xlabel("time (sec)",fontsize=24)
            host.set_ylabel("$v$",fontsize=24)
            par1.set_ylabel(r"$\gamma$",fontsize=24)
            par2.set_ylabel(r"$h$",fontsize=24)
            p1, = host.plot(var[0], var[1],color='blue',linewidth=2.0,label="velocity")
            p2, = par1.plot(var[0][::4], var[2][::4],color='red',linewidth=2.0,label="angle of trajectory")
            p3, = par2.plot(var[0][::6],209*var[3][::6], '--',color='green',linewidth=2.0,label="altitude")
    
            par1.set_ylim(-.15,.05)
            # par2.set_ylim(.008,.02)
            par2.set_ylim(1.5,4.5)
            host.legend(loc='right')
    
            host.axis["left"].label.set_fontsize(18)
            host.axis["left"].label.set_color(p1.get_color())
            par1.axis["right"].label.set_fontsize(18)
            par1.axis["right"].label.set_color(p2.get_color())
            par2.axis["right"].label.set_fontsize(18)
            par2.axis["right"].label.set_color(p3.get_color())
    
            plt.draw()
            plt.show(); plt.clf()
        return
    
    
    plot_reentry_trajectory()
    plot_graph_from_book(soln)
    
solve_using_scikits()
