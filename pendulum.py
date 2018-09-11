import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
matplotlib.rcParams['text.usetex'] = True
from mpl_toolkits.mplot3d import Axes3D


### general pendulum object ###
class Pendulum(object):
    """ The Pendulum object does a few things:  1) it holds the state vector of the
    pendulum, 2) it is responsible for calculating forces that state vector, and 3) it
    is responsible for time-integrating the vector.
    
    At any timestep, an outside function with access to a Pendulum object can ask what its
    state vector is before proceeding with the integration.
    """
    def __init__(self,
                 ### describing the pendulum ###
                 N,
                 init_state = None, # initial state vector, Nx2 array: ([phi_0, phi_d_0],...)
                
                 ### describing the clock ###
                 t0 = 0, # starting time
                 dt = 0.05 # timestep size
                 ):
        
        ### simulation parameters ###
        self.N = N        
        self.init_state = self.get_init_state(init_state)
        self.state = np.copy(self.init_state) # hold onto a copy of the initial state
        self.history = np.array([]) # self.timestep will use this to hold a history
                                    # of the time evolution
        
        
        ### time stuff ###
        self.t0 = t0
        self.dt = dt
        self.t = t0 # the actual clock.
        
        ### physical values ###
        self.l = 9.8 # arm length– you could have an array with a length for every arm, but whatever
        self.g = 9.8 # gravity, duh
    
    def get_init_state(self, init_state):
        """ This initializes some common initial states
        """
        state = np.zeros((self.N, 2))
        if type(init_state) is np.ndarray:
            state = init_state
        
        # you can specify them on the left side
        elif init_state == 'left':
            state[:, 0] = np.zeros(self.N) - (np.pi / 2)
        
        # or to be vertical (helpful for testing)
        elif init_state == 'vertical':
            state[:, 0] = np.zeros(self.N)
        
        # the default is all states horizontal on the viewer's right.
        else:
            state[:, 0] = np.zeros(self.N) + (np.pi / 2)
        
        return(state)
    
    
    ### timestepping and integrating ###
    def integrateRK4(self, state):
        """ time-integration.  This basically just handles the RK4 interstitial
        integration values.  The derivative vector depends on the pendulum itself,
        so that's generated in the pendulum subclasses.
        """
        
        k1 = self.dt * self.get_derivative_vector(state)
        k2 = self.dt * self.get_derivative_vector(state + (k1 / 2))
        k3 = self.dt * self.get_derivative_vector(state + (k2 / 2))
        k4 = self.dt * self.get_derivative_vector(state + k3)

        k = (1 / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

        new_state = state + k      
        return(new_state)
    
    def timestep(self, nsteps, history = None):
        """ timestep the system for nsteps.  This function is responsible for:
        1) call self.integrate to advance the state, 2) advance the clock, 3) record the
        new state in the history, which is returned at the end of the timestepping.
        
        If you have already called self.timestep, you can feed the old history back into
        self.timestep, and it'll append the new states onto it.
        """
        if history == None: # if no history is specified, make an empty array
            history_len = 0 # this tells it where in the history arr to put the next state
            self.history = np.empty((nsteps, 2*self.N))
        
        else: # otherwise, make a new array of the appropriate length
            history_len = nsteps
            new_history_arr = np.empty((history_len + len(history), 2 * self.N))
            new_history_arr[:(len(self.history))] = self.history
            self.history = new_history_arr
            
        
        for step in range(nsteps):
            state = np.copy(self.state) # make a copy so you're not just writing to memory in-place
            self.history[history_len + step] = state.flatten()
            self.state = self.integrateRK4(state) # update the state vector
            self.t += self.dt # advance the clock
        
        return(self.history)

    
    def reset(self):
        """ reset the state to the initial state
        """
        self.state = self.init_state
        print(f'state reset to {self.init_state}')
        return()
    
    ### some troubleshooting functions ###
    def __str__(self):
        """ specifying the object's behavior when called by the built-in print function.
        """
       
        preamble = f"N masses: {self.N} \ntimestep: {int(self.t / self.dt)} \n"
        state_str = self.get_state_str()
        state_str = preamble + state_str
        
        return(state_str)
    
    def get_state_str(self):
        """ formats the state vector into a human-readable table
        """
        str = "       Theta   Theta_d \n"
        for mass in range(len(self.state)):
            str += f"mass{mass}: {self.state[mass, 0]},    {self.state[mass, 1]} \n"
        return(str)

### single and double versions ###
class SinglePendulum(Pendulum):
    """ A subclass of the Pendulum object for a single pendulum.  Takes three extra
    params: b, omega_d and A_d, which are the damping parameter, driving frequency,
    and driving amplitude.  Driving frequency is given as a percentage of the natural
    frequency.
    """
    def __init__(self,
                b = 0, # damping parameter
                omega_d = 0, # driving frequency
                A_d = 0, # driving amplitude
                **kwargs):
        
        Pendulum.__init__(self, N = 1, **kwargs)
        
        self.b = b
        self.omega_d = omega_d
        self.A_d = A_d
        self.omega_0 = np.sqrt(self.g / self.l)
        
    def get_derivative_vector(self, state_n):
        """ state_n is the state vector at timestep n.  It'll usually be the latest state
        vector.
        """
        dot_vec = np.zeros_like(self.state)
        
        Th = state_n[0, 0] # get the theta value
        Th_d = state_n[0, 1] # theta dot
                
        Th_dd = - (self.omega_0 ** 2) * np.sin(Th) # standard EoM
        
        Th_dd += self.A_d * np.cos(self.omega_d * self.omega_0 * self.t) # add driving
        Th_dd -= self.b * Th_d # add damping
        
        dot_vec[0, 0] = Th_d
        dot_vec[0, 1] = Th_dd
        
        return(dot_vec)


class DoublePendulum(Pendulum):
    """ Same as above, but for a double pendulum.  No extra params.
    """
    def __init__(self, **kwargs):
        Pendulum.__init__(self, N = 2, **kwargs)
        
    def get_derivative_vector(self, state_n):
        dot_vec = np.zeros_like(state_n)
        
        ### current state ###
        Th_1 = state_n[0][0]
        Th_d_1 = state_n[0][1]
        Th_2 = state_n[1][0]
        Th_d_2 = state_n[1][1]
        
        ### theta_1 acceleration ###
        a1 = self.g*(np.sin(Th_2)*np.cos(Th_1-Th_2)-2*np.sin(Th_1))
        a2 = -(self.l*Th_d_2*Th_d_2+self.l*Th_d_1*Th_d_1*np.cos(Th_1-Th_2))
        a3 = np.sin(Th_1-Th_2)
        a4 = self.l*(2-np.cos(Th_1-Th_2)*np.cos(Th_1-Th_2))
        
        Th_dd_1 = (a1 + (a2 * a3))/a4
        
        ### theta_2 acceleration ###
        b1 = 2*self.g*(np.sin(Th_1)*np.cos(Th_1-Th_2)-np.sin(Th_2))
        b2 = 2*self.l*Th_d_1*Th_d_1+self.l*Th_d_2*Th_d_2*np.cos(Th_1-Th_2)
        b3 = np.sin(Th_1-Th_2)
        b4 = self.l*(2-np.cos(Th_1-Th_2)*np.cos(Th_1-Th_2))
        
        Th_dd_2 = (b1 + (b2 * b3))/b4
        
        ### derivative vector ###
        dot_vec[0, 0] = Th_d_1
        dot_vec[0, 1] = Th_dd_1
        dot_vec[1, 0] = Th_d_2
        dot_vec[1, 1] = Th_dd_2
        
        return(dot_vec)



### Animating things ###
class Animate(object):
    """ The Animate object takes as input a Pendulum object and the number of time steps
    it should animate for.  It then timesteps the pendulum and reads its state at every
    step to create each frame.
    """
    def __init__(self,
                 pendulum, # a pendulum object
                 nsteps = 10000,
                 phasespace = (0,0) # which position and which momentum should be plotted
                 ):
        
        self.pendulum = pendulum
        self.double = False
        self.check_double()
        self.nsteps = nsteps
        self.phasespace = phasespace
        ratio = 1.5
        
        if self.double:
            self.fig = plt.figure(figsize=(16/ratio,12/ratio))
        else:
            self.fig = plt.figure(figsize=(16/ratio,7/ratio))

        
        if self.double:
            self.ax1 = self.fig.add_subplot(221, aspect = 'equal')
            self.ax2 = self.fig.add_subplot(222, aspect = 'auto')
            self.ax3 = self.fig.add_subplot(212, aspect = 'equal', projection = '3d')
            self.init_torus(self.ax3)
            self.ax3.set_title('configuration space torus')
            self.torus_data_1, self.torus_data_2, self.torus_data_3, = [], [], []
            self.torus_line, = self.ax3.plot(self.torus_data_1,
                                             self.torus_data_2,
                                             self.torus_data_3)
        
        else:
            self.ax1 = self.fig.add_subplot(121, aspect = 'equal')
            self.ax2 = self.fig.add_subplot(122)

        self.ax2xlim = 3
        self.ax2ylim = 3    
        
        self.ax1.set_title('pendulum space')
        self.ax2.set_title(r'phase space for $\theta_{}$ vs $\omega_{}$'.format(self.phasespace[0]+1,
                                                                                self.phasespace[1]+1))   
        
        self.th_data, self.th_d_data, = [], []
        self.rod, = self.ax1.plot(self.th_data, [], animated = True)
        self.mass, = self.ax1.plot(self.th_data, [], 'ro', animated = True)
        
        self.phase_line, = self.ax2.plot(self.th_data, self.th_d_data, animated = True)
        

    def check_double(self):
        if len(self.pendulum.state) == 2:
            self.double = True
            
    def init_plot(self):
        self.ax1.set_ylim(-2.5, 2.5)
        self.ax1.set_xlim(-2.5, 2.5)
        self.ax2.set_xlim(-self.ax2xlim,self.ax2xlim)
        self.ax2.set_ylim(-self.ax2ylim,self.ax2ylim)
        if self.double:
            self.ax3.set_xlim(-5,5)
            self.ax3.set_ylim(-5,5)
            self.ax3.set_zlim(-5,5)
            ret = [self.rod, self.mass, self.phase_line, self.torus_line,]
        else:
            ret = [self.rod, self.mass, self.phase_line,]
        return(ret)

    def init_torus(self, ax):
        ax.clear()
        ax.view_init(30,40)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        return()
    
    def plot_2d_configuration(self):
        th1 = self.pendulum.state[0,0]
        th2 = self.pendulum.state[1,0]
        c, a = 4, 2
        x = (c + a*np.cos(th1)) * np.cos(th2)
        y = (c + a*np.cos(th1)) * np.sin(th2)
        z = a * np.sin(th1)
        self.torus_data_1.append(x)
        self.torus_data_2.append(y)
        self.torus_data_3.append(z)
        return(x,y,z)
        
        

    def get_mass_pos(self):
        xpos = [0]
        ypos = [0]
        for i in range(len(self.pendulum.state)):
            xpos.append(np.sin(self.pendulum.state[i, 0]) + xpos[-1])
            ypos.append(ypos[-1] - np.cos(self.pendulum.state[i, 0]))
        return(xpos, ypos)
        
    def animate(self, i):
        self.pendulum.timestep(1)
        self.th_data.append(self.pendulum.state[self.phasespace[0]][0])
        self.th_d_data.append(self.pendulum.state[self.phasespace[1]][1])
        
        
        # make the phase space axes bigger if the line goes beyond the edge
        xmin, xmax = self.ax2.get_xlim()
        if self.th_data[-1] >  xmax:
            self.ax2.set_xlim(xmin, 2 * xmax)
        
        elif self.th_data[-1] <  xmin:
            self.ax2.set_xlim(2 * xmin, xmax)
        
        elif np.abs(self.th_d_data[-1]) >  self.ax2ylim:
            self.ax2ylim *= 2
            self.ax2.set_ylim(-self.ax2ylim,self.ax2ylim)
        
        xpos, ypos = self.get_mass_pos()
        self.rod.set_data(xpos, ypos)
        self.mass.set_data(xpos, ypos)
        self.phase_line.set_data(self.th_data, self.th_d_data)
        
        if self.double:
            self.plot_2d_configuration()
            self.torus_line.set_data(self.torus_data_1, self.torus_data_2)
            self.torus_line.set_3d_properties(self.torus_data_3)
            ret = [self.rod, self.mass, self.phase_line, self.torus_line, ]
        
        else:
            ret = [self.rod, self.mass, self.phase_line, ]
        
        return(ret)
    
    
    def call_animation(self):
        anim = animation.FuncAnimation(self.fig,
                                       self.animate,
                                       init_func = self.init_plot,
                                       frames = self.nsteps,
                                       interval = 1,
                                       blit = True)
        plt.show()


def animate(pendulum, **kwargs):
    a = Animate(pendulum, **kwargs)
    a.call_animation()


##########################
# example initial states #
##########################

# single simple pendulum, starting above the horizontal
simple = SinglePendulum(init_state = np.array([[3*np.pi/4, 0]]))

# damped, driven pendulum with chaotic parameters
single_chaotic = SinglePendulum(b=0.05,
                                A_d = 0.7,
                                omega_d = 0.6)

# double pendulum starting at horizontal
double_1 = DoublePendulum()

# double pendulum with some funky initial conditions
double_2 = DoublePendulum(init_state = np.array([[3 * np.pi/4, 0],
                                                 [np.pi/2, -0.3]]))

#animate(double_2, phasespace = (1,1))




