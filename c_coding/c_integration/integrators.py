import numpy as np
import functools

###########################
##   NAIVE RK4 STEPPING  ##
###########################
def integrateRK4(state, derivative_function, dt):

        
    k1 = dt * derivative_function(state)
    k2 = dt * derivative_function(state + (k1 / 2))
    k3 = dt * derivative_function(state + (k2 / 2))
    k4 = dt * derivative_function(state + k3)

    k = (1 / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

    new_state = state + k      
    return(new_state, dt)


###############################################
##            ADAPTIVE STEPPING              ##
## after the Cash-Karp RKF45 implementation  ##
## from Numerical Recipes in C by PTVF, 1997 ##
###############################################
def integrateRKF(init_state,
                 derivative_function,
                 init_h,
                 order,
                 eps=1e-12,
                 maxsteps = 1000):
                 
    orders = {'45': stepCK5,
              '78': functools.partial(stepRK8, mode='adaptive')}
    stepfunc = orders[order]
    init_guess, err = stepfunc(init_state, derivative_function, init_h)
    h = init_h
    nsteps = 0
    new_state = np.copy(init_guess)
    
    while np.abs(err) > eps:
    
        h *= ((eps * h) / (2 * err))**(1/4)
        new_state, err = stepfunc(init_state, derivative_function, h)
    
        nsteps += 1
    
        if nsteps == maxsteps:
            raise RuntimeError(f'Could not achieve error of {eps} in {maxsteps} steps.')
            break
    return(new_state, h)
    
def stepCK5(state, derivative_function, h):
    """ The Cash-Karp fifth-order stepper
    returns the 4th-order step along with the error calculated against
    the 5th-order step
    """
    
    
    """A gajillion constants in the butcher tableau.
    """
    
    b21 = 0.2
    b31 = 3.0 / 40.0
    b32 = 9.0 / 40.0
    b41 = 0.3
    b42 = -0.9
    b43 = 1.2
    b51 = -11.0 / 54.0
    b52 = 2.5
    b53 = -70.0 / 27.0
    b54 = 35.0 / 27.0
    b61 = 1631.0 / 55296.0
    b62 = 175.0 / 512.0
    b63 = 575.0 / 13824.0
    b64 = 44275.0 / 110592.0
    b65 = 253.0 / 4096.0
    
    c1 = 37.0 / 378.0
    c3 = 250.0 / 621.0
    c4 = 125.0 / 594.0
    c6 = 512.0 / 1771.0

    dc1 = c1 - (2825.0 / 27648.0)
    dc3 = c3 - (18575.0 / 48384.0)
    dc4 = c4 - (13525.0 / 55296.0)
    dc5 = -277.00 / 14336.0
    dc6 = c6 - 0.25

    """Interstitial steps
    """
    k1 = h * derivative_function(state)
    
    k2 = h * derivative_function(state + (b21 * k1))
    
    k3 = h * derivative_function(state + (b31 * k1) +
                                         (b32 * k2))
                                         
    k4 = h * derivative_function(state + (b41 * k1) +
                                         (b42 * k2) + 
                                         (b43 * k3))
                                         
    k5 = h * derivative_function(state + (b51 * k1) +
                                         (b52 * k2) +
                                         (b53 * k3) +
                                         (b54 * k4))
                                         
    k6 = h * derivative_function(state + (b61 * k1) +
                                         (b62 * k2) +
                                         (b63 * k3) +
                                         (b64 * k4) +
                                         (b65 * k5))
    
    """Weighting the steps to return the next state vector
    """
    new_state = state + h * ((c1 * k1) +
                             (c3 * k3) +
                             (c4 * k4) +
                             (c6 * k6))
    
    """Error by comparison with 5th order step
    """
    new_state_err = np.linalg.norm(h * ((dc1 * k1) +
                                        (dc3 * k3) +
                                        (dc4 * k4) +
                                        (dc5 * k5) +
                                        (dc6 * k6)))
    
    return(new_state, new_state_err)




##########################
##         RK8          ##
## both adaptive and    ##
##  static stepping     ##
##########################
def stepRK8(state, derivative_function, dt, mode = 'adaptive'):
    b21 = 2./27.
    
    b31 = 1./36.
    b32 = 1./12.
    
    b41 = 1./24.
    b43 = 3./24.
    
    b51 = 20./48.
    b53 = -75./48.
    b54 = 75./48.
    
    b61 = 1./20.
    b64 = 5./20.
    b65 = 4./20.
    
    b71 = -25./108.
    b74 = 125./108.
    b75 = -260./108.
    b76 = 250./108.
    
    b81 = 31./300.
    b85 = 61./225.
    b86 = -2./9.
    b87 = 13./900.
    
    b91 = 2.
    b94 = -53./6.
    b95 = 704./45.
    b96 = -107./9.
    b97 = 67./90.
    b98 = 3.
    
    b101 = -91./108.
    b104 = 23./108.
    b105 = -976./135.
    b106 = 311./54.
    b107 = -19./60.
    b108 = 17./6.
    b109 = -1./12.
    
    b111 = 2383./4100.
    b114 = -341./164.
    b115 = 4496./1025.
    b116 = -301./82.
    b117 = 2133./4100.
    b118 = 45./82.
    b119 = 45./164.
    b1110 = 18./41.
    
    b121 = 3./205.
    b126 = -6./41.
    b127 = -3./205.
    b128 = -3./41.
    b129 = 3./41.
    b1210 = 6./41.
    
    b131 = -1777./4100.
    b134 = -341./164.
    b135 = 4496./1025.
    b136 = -289./82.
    b137 = 2193./4100.
    b138 = 51./82.
    b139 = 33./164.
    b1310 = 12./41.
    b1312 = 1.

    c1 = 41./840.
    c6 = 34./105.
    c7 = 9./35.
    c8 = 9./35.
    c9 = 9./280.
    c10 = 9./280.
    c12 = 9./280.
    c13 = 41./840.
    
    k1 = dt * derivative_function(state)

    k2 = dt * derivative_function(state + (b21 * k1))
    
    k3 = dt * derivative_function(state + (b31 * k1) + 
                                          (b32 * k2))
    
    k4 = dt * derivative_function(state + (b41 * k1) +
                                          (b43 * k3))
    
    k5 = dt * derivative_function(state + (b51 * k1) +
                                          (b53 * k3) +
                                          (b54 * k4))
    
    k6 = dt * derivative_function(state + (b61 * k1) + 
                                          (b64 * k4) + 
                                          (b65 * k5))

    k7 = dt * derivative_function(state + (b71 * k1) +
                                          (b74 * k4) +
                                          (b75 * k5) +
                                          (b76 * k6))
    
    k8 = dt * derivative_function(state + (b81 * k1) +
                                          (b85 * k5) +
                                          (b86 * k6) +
                                          (b87 * k7))
    
    k9 = dt * derivative_function(state + (b91 * k1) +
                                          (b94 * k4) +
                                          (b95 * k5) +
                                          (b96 * k6) +
                                          (b97 * k7) +
                                          (b98 * k8))
  
    k10 = dt * derivative_function(state + (b101 * k1) +
                                           (b104 * k4) +
                                           (b105 * k5) +
                                           (b106 * k6) +
                                           (b107 * k7) +
                                           (b108 * k8) +
                                           (b109 * k9))

    k11 = dt * derivative_function(state + (b111 * k1) +
                                           (b114 * k4) +
                                           (b115 * k5) +
                                           (b116 * k6) +
                                           (b117 * k7) +
                                           (b118 * k8) +
                                           (b119 * k9) +
                                           (b1110 * k10))

    k12 = dt * derivative_function(state + (b121 * k1) +
                                           (b126 * k6) +
                                           (b127 * k7) +
                                           (b128 * k8) +
                                           (b129 * k9) +
                                           (b1210 * k10))

    k13 = dt * derivative_function(state + (b131 * k1) +
                                           (b134 * k4) +
                                           (b135 * k5) +
                                           (b136 * k6) +
                                           (b137 * k7) +
                                           (b138 * k8) +
                                           (b139 * k9) +
                                           (b1310 * k10) +
                                           (b1312 * k12))


    k = dt * ((c6 * k6) +
              (c7 * k7) +
              (c8 * k8) +
              (c9 * k9) +
              (c10 * k10) +
              (c12 * k12) +
              (c13 * k13))

    new_state = state + k
    
    
    if mode == 'static':
        retvals = [new_state, dt]
        
    elif mode == 'adaptive':
        err_factor = -41. / 840.
        new_state_err = np.linalg.norm(err_factor * dt * (k1 + k11 - k12 - k13))
        retvals = [new_state, new_state_err]
        
    return(retvals)


integrator_dict = {'RK4': integrateRK4,
                   'RK8': functools.partial(stepRK8, mode = 'static'),
                   'RKF45': functools.partial(integrateRKF, order = '45'),
                   'RKF78': functools.partial(integrateRKF, order = '78')}

    