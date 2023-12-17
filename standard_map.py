import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time

def driver():
    '''
    Function parameters to change:
    (x0, y0) - initial condition for a specific trajectory
    N - Number of iterations for each initial condition
    k - parameter
    num_points - the number of points in both x and y to plot as initial conditions to fill out the map
    manifold_appx_dist - how far from the fixed point along the eigenvectors to begin iterating to approximate the manifolds
    '''
    x0 = .2
    y0 = .3
    N = 100
    k = (np.pi/2)**2
    num_points = 25
    manifold_appx_dist = 1e-10

    # plots a single trajectory in the standard map starting from (x0, y0)
    # plot_trajectory(x0, y0, N, k, '.') 
 
    # plots general picture of entire map, optionally plots separatrix, manifold approximation, and a particular intial value
    # see function def for details
    plot_map(N, k, num_points, x0, y0, manifold_appx_dist, init_val_on = True, sep_on = True, manifold_on = True, point_type = ',', a = 0.5, b = 0.5)

def plot_map(N, k, num_points, x_init, y_init, manifold_appx_dist, init_val_on = False, sep_on = False, manifold_on = False, point_type = ',', a = 0.5, b = 0.5):
    '''
    Plots num_points^2 intial conditions to get a general picture of the behavior of the standard map.
    Also plots the following (though these can be turned on or off with the boolean inputs):
    - plots a paticular trajectory with (x_init, y_init) initial condition
    - plots the separatrix of the (0,0) resonance/fixed point
    - approximation of the manifolds (the approximation is bad for large k but still interesting)
    Input:
    - N - number of iterations for each initial point
    - k - parameter
    - num_points - number of x and y values to use as initial conditions. Total of num_points^2 initial conditions fill out the map
    - x_init, y_init - starting x and y value of specific initial condition
    - manifold_appx_dist - how far from the fixed point along the eigenvectors to begin iterating to approximate the manifolds
    - init_val_on, sep_on, manifold_on - booleans turn the initial value, separatrix, and manifold plots on or off, default is off
    - point_type - point style to plot the main map with
    - a - x value max of plot
    - b - y value max of plot
    Output: plot of standard map
    '''

    x0 = np.linspace(-a, a, num_points)
    y0 = np.linspace(-b, b, num_points)
    if manifold_on:
        eigvals, eigvecs = Df_eigs(k, -0.5)
        eigvecs[:, 0] = normalize(eigvecs[:, 0])
        eigvecs[:, 1] = normalize(eigvecs[:, 1])
    if sep_on:
        y_separatrix = (1/np.pi)*np.sqrt(k/2 + (k/2)*np.cos(2*np.pi*x0))
        # y_separatrix_2 = .5*np.sqrt(1 + (k/(np.pi))**2 + ((k/(np.pi))**2)*np.cos(4*np.pi*x0))
    random.seed(3) # for colors
    plt.figure()    

    colors = ['#CC3366','#FF0066','#CC6699','#CC00FF','#9933FF','#9900FF','#00FFCC','#00FF33','#33CC00','#00CC99','#33CC99','#33FFFF','#33CCCC','#3399CC','#0000CC','#0099FF','#6633FF','#66CC99','#33CC66','#66CC00','#33FF33','#FF3333','#CC0033','#0000FF','#00CCFF','#CC0000']
    for i in range(num_points):
        for j in range(num_points):
            x_traj, y_traj = initial_condition_trajectory(x0[i], y0[j], k, N)
            plt.plot(x_traj,y_traj, point_type, color = random.choice(colors))
    c1 = 'blue'
    c2 = 'blue'
    if manifold_on:
        origin = np.array([[-.5, -.5], [0, 0]])
        if eigvals[0] > 1:
            c1 = 'red'
        if eigvals[1] > 1:
            c2 = 'red'
        # plt.quiver(*origin, eigvecs[:, 0], eigvecs[:, 1], color=[c1, c2], scale=12) # plots eigenvectors
    if init_val_on:
        x_traj, y_traj = initial_condition_trajectory(x_init, y_init, k, N)
        lbl = 'Trajectory from ('+str(x_init)+','+str(y_init)+')'
        plt.plot(x_traj,y_traj,'.', color = 'b', label = lbl)
        plt.plot(x_traj[0],y_traj[0], 'ro', label = 'starting point')
        plt.plot(x_traj[N-1],y_traj[N-1], 'co', label = 'ending point')
    if sep_on:
        plt.plot(x0, y_separatrix, 'k,-', label='separatrix')
        plt.plot(x0, -y_separatrix, 'k,-')
        # plt.plot(x0, y_separatrix_2, 'k,-', label='secondary separatrix')
        # plt.plot(x0, -y_separatrix_2+1, 'k,-')
    if manifold_on:
        x_man, y_man = iter_manifold(eigvecs, 0, 0.5, 0, manifold_appx_dist)
        x_traj, y_traj = initial_condition_trajectory(x_man, y_man, k, 5*N)
        lbl = 'Manifold approximation'
        plt.plot(x_traj,y_traj,'m.', label = lbl)
        x_man, y_man = iter_manifold(eigvecs, 1, 0.5, 0, manifold_appx_dist)
        x_traj, y_traj = initial_condition_trajectory(x_man, y_man, k, 5*N)
        plt.plot(x_traj,y_traj,'m.')
    plt.xlim(-a, a)
    plt.ylim(-b, b)
    plt.title('Standard Map with '+str(N)+' Iterations, k='+str(k))
    plt.legend(loc='lower right')
    plt.show()

def plot_trajectory(x0, y0, N, k, point_type = '.'):
    '''
    Plots a single trajectory in the standard map starting from (x0, y0)
    Input:
    - (x0, y0) - intial condition
    - N - number of iterations for each initial point
    - k - parameter
    - point_type - type of point used for plot
    Output: Plot of single trajectory
    '''
    x_traj, y_traj = initial_condition_trajectory(x0, y0, k, N)
    plt.figure()    
    plt.plot(x_traj,y_traj,'b'+point_type, label= 'trajectory')
    lbl = 'start ('+str(x0)+','+str(y0)+')'
    plt.plot(x_traj[0],y_traj[0], 'ro', label = lbl)
    plt.plot(x_traj[N-1],y_traj[N-1], 'co', label = 'ending point')
    
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.title('Standard Map with '+str(N)+' Iterations, k='+str(k))
    plt.show()   

def mod1(x):
    # mods the x value to make the map correctly periodic (built in mod does not work)
    x_mod1 = x - np.floor(x+.5) 
    return x_mod1

def chirikov(x, y, k):
    # One iteration of Chirikov's standard map with parameter k
    # note: x is generally p, y is generally q for Hamiltonian
    y_prime = y - (k/(2*np.pi))*np.sin(2*np.pi*x)
    x_prime = mod1(x + y_prime)
    
    return x_prime, y_prime

def initial_condition_trajectory(x0, y0, k, N):
    # find the trajectory of intial condition (x0, y0)
    ''' 
    Input: x0 - intial x, y0 - initial y, k - parameter, N - number of iterations 
    Output: first N points (x and y values) of the trajectory starting at (x0, y0)
    '''
    x_traj = np.zeros(N)
    y_traj = np.zeros(N)
    x_traj[0], y_traj[0] = x0, y0
    for j in range(1,N):
        x_traj[j], y_traj[j] = chirikov(x_traj[j-1], y_traj[j-1], k)

    return x_traj, y_traj

def iter_manifold(norm_eigvecs, vec_col, x0, y0, L = 1e-10):
    # go in direction of stable and unstable eigenvector from saddle point
    '''
    Input:
    - norm_eigvecs - normalized eigenvectors (2x2 matrix with columns as eigenvectors) at the fixed point
    - vec_col - which column (eigenvector) of norm_eigvector to use
    - (x0, y0) - saddle point
    - L - distance along eigenvector to iterate from (x0, y0)
    Output:
    - point distance L away from saddle point along eigenvector
    '''
    v1 = norm_eigvecs[0,vec_col]
    v2 = norm_eigvecs[1,vec_col]
    x_prime = x0 + L*v1
    y_prime = y0 + L*v2
    return x_prime, y_prime

def Df_eigs(k, x0):
    '''
    Finds eigenvalues and eigenvectors of Df, the jacobian of the standard map at an intial condition x0
    Input:
    - k - parameter
    - x0 - x value of point to evaluate eigenpairs at (y-value has no effect in this case)
    Output:
    - eigenvalue vector, length 2
    - eigenvectors, 2x2 matrix with the jth column being the eigenvector corresponding to the jth entry of the eigenvalue vector. 2x2 in this case
    '''

    kcos = k*np.cos(2*np.pi*x0)
    Df = np.zeros((2,2))
    Df[0,0] = 1-kcos
    Df[0,1] = 1
    Df[1,0] = -kcos
    Df[1,1] = 1
    # tr = 2 - kcos # Trace: not used in this but can give some analytical information, so I left it in
    
    '''
    Eigenvalues calculated more analytically - probably would be better than built in eig but for 2x2 shouldn't really matter.
    I would probably use these to then find the eigenvectors if I continued working on this code
    tr = 2 - kcos
    discrim = np.sqrt(tr**2 - 4)
    eigvals = [(tr - discrim)/2, (tr + discrim)/2]
    '''

    return la.eig(Df)

def normalize(v):
    '''
    normalizes a vector
    - should account for zero vector, could almost certainly be done better, but should be sufficient for this use at least
    '''
    norm = la.norm(v)
    if norm < 1e-16: 
       return v
    return v / norm

driver()

