#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

static double f1(double x, double y);
static double f2(double x, double y);

/*
integrate_rk4: a fourth-order Runge-Kutta integrator

Arguments:
    t0  The clock start time
    N   Number of timesteps
    dt  Timestep
    init_state  An array with [x_0, xd_0]
    history   The array which will hold the whole integration
      looks like: [[x_0, xd_0], ... [x_N, xd_N]]
    
The function returns 0 and edits the history array in-place.  Not sure
if this is the best way to do this.
*/
int * integrate_rk4(double t0,
                    int N,
                    double dt,
                    double init_state[2][1], 
                    double history[2][N]){
    
    /* RK4 coefficients */
    double k1, k2, k3, k4;
    
    int step;
    double T = N * dt;
    double t = t0;
    
    /* copy init_state into another array which will be updated at each timestep */
    double current_state[2][1];
    current_state[0][0] = init_state[0][0];
    current_state[1][0] = init_state[1][0];
    
    /* position, velocity */
    double x1, xd1, new_x1, new_x1d, thing;
    
    /* do the integration */
    for(step = 0; step < N; step++){
        x1 = current_state[0][0];
        xd1 = current_state[1][0];
        
        k1 = dt * f1(t, x1);
        k2 = dt * f1(t + dt/2, x1);
        k3 = dt * f1(t + dt/2, x1);
        k4 = dt * f1(t + dt, x1);

        
        t += dt;
        new_x1 = x1 + ((1./6.) * (k1 + (2 * k2) + (2 * k3) + k4));
        new_x1d = f1(t, x1);
        current_state[0][0] = new_x1;
        current_state[1][0] = new_x1d;
        history[0][step] = new_x1;
        history[1][step] = new_x1d;
    }

    return 0;
}

static double f1(double x, double y){
    return(sin(x - M_PI/2));
    }

static double f2(double x, double y){
    return(x * y);
}


