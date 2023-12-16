import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
from cycler import cycler


def driver():

    x0 = -.5
    y0 = .5
    N = 500
    k = .1
    num_points = 50
    manifold_appx_dist = 1e-10

    # origin = np.array([[-.5, -.5], [0, 0]])


    # eigvecs[:, 1] = eigvecs[:, 1]/la.norm(eigvecs[:, 1])
    '''
    print('eigenpair 1:')
    print('eigenvalue:', eigvals[0])
    print(eigvecs[:,0])
    print()
    print('eigenpair 2:')
    print('eigenvalue:', eigvals[1])
    print(eigvecs[:,1])
    '''
    x_traj, y_traj = initial_condition_trajectory(x0, y0, k, N)
    
    #plot_trajectory(x_traj, y_traj, N,'.')

    t_s = time.time()  
    plot_map(N, k, num_points, x0, y0, manifold_appx_dist, init_val = True, sep_on = True, point_type = ',', a = 0.5, b = 0.5, eigvecs_on = True, manifold_on = True)
    t_l = time.time()
    t = t_l - t_s
    print('time to plot Map:', t)

def plot_trajectory(x_traj, y_traj, N, point_type = '.'):

    plt.figure()    
    plt.plot(x_traj,y_traj,'b'+point_type, label= 'trajectory')
    plt.plot(x_traj[0],y_traj[0], 'ro', label = 'starting point')
    plt.plot(x_traj[N-1],y_traj[N-1], 'co', label = 'ending point')
    
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()   

def plot_map(N, k, num_points, x_init, y_init, manifold_appx_dist, init_val = False, sep_on = False, point_type = ',', a = 0.5, b = 0.5,  eigvecs_on = False, manifold_on = False):
    x0 = np.linspace(-a, a, num_points)
    y0 = np.linspace(-b, b, num_points)
    if eigvecs_on:
        eigvals, eigvecs = Df_eigs(k, -0.5)
        eigvecs[:, 0] = normalize(eigvecs[:, 0])
        eigvecs[:, 1] = normalize(eigvecs[:, 1])

    if sep_on:
        y_separatrix = (1/np.pi)*np.sqrt(k/2 + (k/2)*np.cos(2*np.pi*x0))
        y_separatrix_2 = .5*np.sqrt(1 + (k/(np.pi))**2 + ((k/(np.pi))**2)*np.cos(4*np.pi*x0))

    random.seed(2)
    plt.figure()    

    colors = ['#CC3366','#FF0066','#CC6699','#CC00FF','#9933FF','#9900FF','#00FFCC','#00FF33','#33CC00','#00CC99','#33CC99','#33FFFF','#33CCCC','#3399CC','#0000CC','#0099FF','#6633FF','#66CC99','#33CC66','#66CC00','#33FF33','#FF3333','#CC0033','#0000FF','#00CCFF','#CC0000']
    for i in range(num_points):
        for j in range(num_points):
            x_traj, y_traj = initial_condition_trajectory(x0[i], y0[j], k, N)
            plt.plot(x_traj,y_traj, point_type, color = random.choice(colors))
    c1 = 'blue'
    c2 = 'blue'
    if eigvecs_on:
        origin = np.array([[-.5, -.5], [0, 0]])
        if eigvals[0] > 1:
            c1 = 'red'
        if eigvals[1] > 1:
            c2 = 'red'
        # plt.quiver(*origin, eigvecs[:, 0], eigvecs[:, 1], color=[c1, c2], scale=12)
    if init_val:
        x_traj, y_traj = initial_condition_trajectory(x_init, y_init, k, N)
        lbl = 'Trajectory from ('+str(x_init)+','+str(y_init)+')'
        plt.plot(x_traj,y_traj,'.', color = 'b', label = lbl)
        plt.plot(x_traj[0],y_traj[0], 'ro', label = 'starting point')
        plt.plot(x_traj[N-1],y_traj[N-1], 'co', label = 'ending point')
    if sep_on:
        plt.plot(x0, y_separatrix, 'k,-', label='separatrix')
        plt.plot(x0, -y_separatrix, 'k,-')
        '''plt.plot(x0, y_separatrix_2, 'k,-', label='secondary separatrix')
        plt.plot(x0, -y_separatrix_2+1, 'k,-')'''
    if manifold_on and eigvecs_on:
        x_man, y_man = iter_manifold(eigvals, eigvecs, 0, 0.5, 0, manifold_appx_dist)
        x_traj, y_traj = initial_condition_trajectory(x_man, y_man, k, 5*N)
        lbl = 'Manifold approximation'
        plt.plot(x_traj,y_traj,'m.', label = lbl)
        x_man, y_man = iter_manifold(eigvals, eigvecs, 1, 0.5, 0, manifold_appx_dist)
        x_traj, y_traj = initial_condition_trajectory(x_man, y_man, k, 5*N)
        plt.plot(x_traj,y_traj,'m.')
    plt.xlim(-a, a)
    plt.ylim(-b, b)
    plt.title('Standard Map with '+str(N)+' Iterations, k='+str(k))
    plt.legend(loc='lower right')
    plt.show()

def point_color(x,y,x_max,y_max):
    # The original colour of a point is uniquely determined by an RGB value determined by
    # the starting point where Red is given by P0, Blue is given by Q0, and
    # Green is given by P0+Q0.
    # print(str(x)+','+str(y))
    return (abs(x/x_max), abs(y/y_max), 0)

def mod1(x):
    x_mod1 = x - np.floor(x+.5) 
    return x_mod1

# x is generally p, y is generally q for Hamiltonian
def chirikov(x, y, k):
    # One iteration of Chirikov's standard map with parameter k
    y_prime = y - (k/(2*np.pi))*np.sin(2*np.pi*x)
    x_prime = mod1(x + y_prime)
    
    return x_prime, y_prime

def initial_condition_trajectory(x0, y0, k, N):
    # find the trajectory of intial condition (x0, y0)
    ''' input: x0 - intial x, y0 - initial y, k - parameter, N - number of iterations '''

    # calculate the first N points (x and y values) of the trajectory starting at (x0, y0)
    x_traj = np.zeros(N)
    y_traj = np.zeros(N)
    x_traj[0], y_traj[0] = x0, y0
    for j in range(1,N):
        x_traj[j], y_traj[j] = chirikov(x_traj[j-1], y_traj[j-1], k)

    return x_traj, y_traj

#def plot_trajectory(x_traj, y_traj, N, m):
    # Plots the trajectories of each intial condition
    # Input: x_traj - m x N matrix with each column containing the x values for a single trajectory (i.e for a given intial condition)
    # y_traj - m x N matrix with each column containing the y values for a single trajectory (i.e for a given intial condition)
    # N - the number of points in each trajectory (i.e. # iterations of (x0, y0)). Also the number of columns of the matrices x_traj and y_traj
    # m - number of different trajectories. Also the rows of the matrices

def iter_manifold(eigvals, norm_eigvecs, vec_col, x0, y0, L = 1e-10):
    # go in direction of stable and unstable eigenvector from saddle point
    v1 = norm_eigvecs[0,vec_col]
    v2 = norm_eigvecs[1,vec_col]
    x_prime = x0 + L*v1
    y_prime = y0 + L*v2
    return x_prime, y_prime


def Df_eigs(k, x0):
    # returns eigenvalues of Df, the jacobian of the standard map at an intial condition x0
    '''
    kcos = k*np.cos(2*np.pi*x0)
    Df = np.zeros((2,2))
    Df[0,0] = 1-kcos
    Df[0,1] = 1
    Df[1,0] = -kcos
    Df[1,1] = 1
    tr = 2 - kcos
    discrim = np.sqrt(tr**2 - 4)
    eigvals = [(-tr - discrim)/2, (-tr + discrim)/2]
    vec = la.solve(Df)

    return eigvals '''
    kcos = k*np.cos(2*np.pi*x0)
    Df = np.zeros((2,2))
    Df[0,0] = 1-kcos
    Df[0,1] = 1
    Df[1,0] = -kcos
    Df[1,1] = 1
    tr = 2 - kcos
    
    tr = 2 - kcos
    discrim = np.sqrt(tr**2 - 4)
    eigvals = [(tr - discrim)/2, (tr + discrim)/2]
    
    '''A1 = Df
    A2 = Df
    A1[0,0] = A1[0,0] - eigvals[0]
    A1[1,1] = A1[1,1] - eigvals[0]

    A2[0,0] = A2[0,0] - eigvals[1]
    A2[1,1] = A2[1,1] - eigvals[1]'''


    # print(la.eig(Df))
    return la.eig(Df)

def normalize(v):
    norm = la.norm(v)
    if norm < 1e-16: 
       return v
    return v / norm

driver()

