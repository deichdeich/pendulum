import pend_integrate_wrapper as piw
import pendulum
import numpy as np
import matplotlib.pyplot as plt


def get_equal_E_state(Ee, Th2dd):
    partial_state = np.zeros(4, dtype=np.float64)
    partial_state[3] = Th2dd
#    print(partial_state)
    pend = pendulum.DoublePendulum(init_state = partial_state.reshape((2,2)))
    ham = pend.H
#    print(ham)
    i = 0
    while abs(ham - Ee) > 1e-12:
        if ham > Ee:
            fact = -1
        else:
            fact = 1
            
        diff = abs(ham-Ee)
        
        partial_state[1] += fact * diff/1000
        pend = pendulum.DoublePendulum(init_state = partial_state.reshape((2,2)))
        ham = pend.H
        i+=1
        if i>1000:
            raise RuntimeError("conditions didn't converge")
                  
    return(partial_state)

def poin(data):
    th1 = data[:,1]
    th1d = data[:,2]
    th2 = data[:,3]
    th2d = data[:,4]
    th1_good = np.where(np.abs(th1) < 0.001)
    th1d_good = np.where(th1d > 0)
    poininds = np.intersect1d(th1_good, th1d_good)
    poinpoints = np.zeros((len(poininds), 2))
    poinpoints[:,0] = th2[poininds]
    poinpoints[:,1] = th2d[poininds]
    return(poinpoints)

def generate_states(nstates, Th2dd_range = (-1.5,1), energy = -50):
    Th2dd_min, Th2dd_max = Th2dd_range
    inits = np.linspace(Th2dd_min,Th2dd_max,nstates)
    states = []
    i = 0
    while len(states) < nstates:
        try:
            x = get_equal_E_state(-50, inits[i])
            states.append(x)
            i+=1
        except:
            pass
    return(states)
    
def generate_data(states, n_steps, **kwargs):
    i = 0
    data = np.zeros((len(states), n_steps, 5))
    print(f'Generating data from {len(states)} initial states...')
    for state in states:
        print(f'\t state {i+1}...', end='', flush=True)
        statedata = piw.rkf78(nsteps = n_steps, tol=1e-8, y0=state, **kwargs)
        data[i] = statedata
        print(' done.')
        i+=1
    return(data)
    
def get_section(nstates):
    states = generate_states(nstates, (-1.9,1.5))
    data = generate_data(states, 10000, tmax = 10)
    poinplot = plt.figure()
    poinplot.ax1 = poinplot.add_subplot(111)
    for i in range(len(data)):
        statedata = data[i]
        poindata = poin(statedata)
        th = poindata[:,0]%(2*np.pi)
        th[th > np.pi] -= 2*np.pi 
        poinplot.ax1.scatter(th, poindata[:,1], s=1, c='blue')
    return(poinplot)
    