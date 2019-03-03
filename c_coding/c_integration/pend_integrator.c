#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <math.h>
#include <gsl/gsl_vector.h>


#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.

#define GRAVITY 10.

double _rk78(double (*f)(double, double), double *y, double x, double h);

int _stepper(double (*f)(double, double),
             double y[],
             double x,
             double h,
             double xmax,
             double *h_next,
             double tolerance);

double _single_step(double (*f)(double, double),
                    double *y,
                    double x0,
                    double h);

gsl_vector *_vect_add(int vect_len, int count, ...);
int _populate_history(int hist_len,
                      double history[hist_len][5],
                      int step,
                      double clock,
                      gsl_vector *state);

//////////////////////////////////////////////////////////////////////
//     rk4: a fourth-order Runge-Kutta integrator for scalar ODE's 
//
//  Arguments:
//        t0  The clock start time
//        N   Number of timesteps
//        dt  Timestep
//        init_state  An array with [x_0, xd_0]
//        history   The array which will hold the whole integration
//                    looks like: [[x_0, xd_0], ... [x_N, xd_N]]
//
//
/////////////////////////////////////////////////////////////////////
int * rk4(double t0,
          double (*f)(double, double),
          int N,
          double dt,
          double init_state[2][1], 
          double history[2][N]){
    
    /* RK4 coefficients */
    double k1, k2, k3, k4;
    
    int step;
    double t = t0;
    
    /* copy init_state into another array which will be updated at each timestep */
    double current_state[2][1];
    current_state[0][0] = init_state[0][0];
    current_state[1][0] = init_state[1][0];
    
    /* position, velocity */
    double x1, xd1, new_x1, new_x1d;
    
    /* do the integration */
    for(step = 0; step < N; step++){
        x1 = current_state[0][0];
        xd1 = current_state[1][0];
        
        k1 = dt * (*f)(t, x1);
        k2 = dt * (*f)(t + dt/2, x1);
        k3 = dt * (*f)(t + dt/2, x1);
        k4 = dt * (*f)(t + dt, x1);

        
        t += dt;
        new_x1 = x1 + ((1./6.) * (k1 + (2 * k2) + (2 * k3) + k4));
        new_x1d = (*f)(t, x1);
        current_state[0][0] = new_x1;
        current_state[1][0] = new_x1d;
        history[0][step] = new_x1;
        history[1][step] = new_x1d;
    }
    return 0;
}

////////////////////////////////////////////////////////////////////
//  rkf78: an embedded Runge-Kutta-Felberg method
//
// Arguments
//    *f            pointer to the function which gives the derivative of the system
//    xmin          The starting point along the integration coordinate
//    xmax          The ending point
//    init_state    The inish condish at xmin: [Th1, Th2, Th1_d, Th2_d]
//    hist_len      The total number of timesteps (and length of the history array)
//    history       An array of size (5,N) containing all integration steps, edited in-place
//    tolerance     The tolerance for each step
//    h             Initial step size
/////////////////////////////////////////////////////////////////////////
int rkf78(double (*f)(double, double),
          double xmin,
          double xmax,
          double init_state[4],
          int hist_len,
          double history[hist_len][5],
          double tol,
          double h){
    
    double stepsize = (xmax - xmin) / hist_len;
    double hpt;
    
    double x1 = xmin;
    double x2 = xmin + stepsize;
    
    gsl_vector *init_state_vec = gsl_vector_alloc(4);
    _arr2vec(4, init_state, init_state_vec);
    
    gsl_vector *out_state_vec = gsl_vector_alloc(4)
    
    for (int i = 0; i < hist_len; i++){
        _stepper(f, init_state_vec, out_state_vec, x1, h, x2, &hpt, tol);
        _populate_history(hist_len, history, step, x2, out_state_vec);
        x1 = x2;
        x2 += stepsize;
        gsl_vector_memcpy(init_state_vec, out_state_vec);
    }
    return 0;

}

int _stepper(double (*f)(double, double),
             gsl_vector *in_state_vec,
             gsl_vector *out_state_vec,
             double x,
             double h,
             double xmax,
             double *h_next,
             double tolerance ){

   static const double err_exponent = 1.0 / 7.0;

   double scale;
   gsl_vector *temp_in_vec = gsl_vector_alloc(4);
   gsl_vector *temp_out_vec = gsl_vector_alloc(4);
   double err;
   double yy;
   int i;
   int last_interval = 0;

      // Verify that the step size is positive and that the upper endpoint //
      // of integration is greater than the initial enpoint.               //

   if (xmax < x || h <= 0.0) return -2;
   
       // If the upper endpoint of the independent variable agrees with the //
       // initial value of the independent variable.  Set the value of the  //
       // dependent variable and return success.                            //

   *h_next = h;
   gsl_vector_memcpy(out_state_vec, in_state_vec);
   if (xmax == x) return 0; 

       // Insure that the step size h is not larger than the length of the //
       // integration interval.                                            //
  
   if (h > (xmax - x) ) { h = xmax - x; last_interval = 1;}

        // Redefine the error tolerance to an error tolerance per unit    //
        // length of the integration interval.                            //

   tolerance /= (xmax - x);

        // Integrate the diff eq y'=f(x,y) from x=x to x=xmax trying to  //
        // maintain an error less than tolerance * (xmax-x) using an     //
        // initial step size of h and initial value: y = y[0]            //

   gsl_vector_memcpy(temp_in_vec, in_state_vec);
   while ( x < xmax ) {
      scale = 1.0;
      for (i = 0; i < ATTEMPTS; i++) {
         
         err = fabs(_single_step(f,
                                 temp_in_vec,
                                 temp_out_vec,
                                 x,
                                 h));
         
         if (err == 0.0) {
         scale = MAX_SCALE_FACTOR;
         break;
         }
         
         double statenorm = gsl_blas_dnrm2(temp_in_vec);
         
         yy = (statenorm == 0.0) ? tolerance : statenorm;
         
         scale = 0.8 * pow( tolerance * yy /  err , err_exponent );
         scale = min( max(scale,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);
         
         if ( err < ( tolerance * yy ) ) break;
         
         h *= scale;
         
         if ( x + h > xmax ) h = xmax - x;
         
         else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
      }
      if ( i >= ATTEMPTS ) {
            *h_next = h * scale;
            return -1;
            }
      
      gsl_vector_memcpy(temp_in_vec, temp_out_vec);
      x += h;
      h *= scale;
      *h_next = h;
      
      if ( last_interval ) break;
      
      if (  x + h > xmax ) { last_interval = 1; h = xmax - x; }
      
      else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
   }
   gsl_vector_memcpy(out_state_vec, temp_out_vec);
   return 0;
}



double _single_step(double (*f)(double, gsl_vector),
                    gsl_vector *in_vec,
                    gsl_vector *out_vec,
                    double x0,
                    double h){

    const double c_1_11 = 41.0 / 840.0;
    const double c6 = 34.0 / 105.0;
    const double c_7_8= 9.0 / 35.0;
    const double c_9_10 = 9.0 / 280.0;

    const double a2 = 2.0 / 27.0;
    const double a3 = 1.0 / 9.0;
    const double a4 = 1.0 / 6.0;
    const double a5 = 5.0 / 12.0;
    const double a6 = 1.0 / 2.0;
    const double a7 = 5.0 / 6.0;
    const double a8 = 1.0 / 6.0;
    const double a9 = 2.0 / 3.0;
    const double a10 = 1.0 / 3.0;

    const double b31 = 1.0 / 36.0;
    const double b32 = 3.0 / 36.0;
    const double b41 = 1.0 / 24.0;
    const double b43 = 3.0 / 24.0;
    const double b51 = 20.0 / 48.0;
    const double b53 = -75.0 / 48.0;
    const double b54 = 75.0 / 48.0;
    const double b61 = 1.0 / 20.0;
    const double b64 = 5.0 / 20.0;
    const double b65 = 4.0 / 20.0;
    const double b71 = -25.0 / 108.0;
    const double b74 =  125.0 / 108.0;
    const double b75 = -260.0 / 108.0;
    const double b76 =  250.0 / 108.0;
    const double b81 = 31.0/300.0;
    const double b85 = 61.0/225.0;
    const double b86 = -2.0/9.0;
    const double b87 = 13.0/900.0;
    const double b91 = 2.0;
    const double b94 = -53.0/6.0;
    const double b95 = 704.0 / 45.0;
    const double b96 = -107.0 / 9.0;
    const double b97 = 67.0 / 90.0;
    const double b98 = 3.0;
    const double b10_1 = -91.0 / 108.0;
    const double b10_4 = 23.0 / 108.0;
    const double b10_5 = -976.0 / 135.0;
    const double b10_6 = 311.0 / 54.0;
    const double b10_7 = -19.0 / 60.0;
    const double b10_8 = 17.0 / 6.0;
    const double b10_9 = -1.0 / 12.0;
    const double b11_1 = 2383.0 / 4100.0;
    const double b11_4 = -341.0 / 164.0;
    const double b11_5 = 4496.0 / 1025.0;
    const double b11_6 = -301.0 / 82.0;
    const double b11_7 = 2133.0 / 4100.0;
    const double b11_8 = 45.0 / 82.0;
    const double b11_9 = 45.0 / 164.0;
    const double b11_10 = 18.0 / 41.0;
    const double b12_1 = 3.0 / 205.0;
    const double b12_6 = - 6.0 / 41.0;
    const double b12_7 = - 3.0 / 205.0;
    const double b12_8 = - 3.0 / 41.0;
    const double b12_9 = 3.0 / 41.0;
    const double b12_10 = 6.0 / 41.0;
    const double b13_1 = -1777.0 / 4100.0;
    const double b13_4 = -341.0 / 164.0;
    const double b13_5 = 4496.0 / 1025.0;
    const double b13_6 = -289.0 / 82.0;
    const double b13_7 = 2193.0 / 4100.0;
    const double b13_8 = 51.0 / 82.0;
    const double b13_9 = 33.0 / 164.0;
    const double b13_10 = 12.0 / 41.0;

    const double err_factor  = -41.0 / 840.0;

    gsl_vector *k1, *k2, *k3, *k4, *k5, *k6, *k7, *k8, *k9, *k10, *k11, *k12, *k13;
    
    double h2_7 = a2 * h;

    k1 = (*f)(x0, *y);
    k2 = (*f)(x0 + h2_7, *y + h2_7 * k1);
    
    k3 = (*f)(x0 + a3*h, *y + h * (b31*k1 +
                                   b32*k2));

    k4 = (*f)(x0 + a4*h, *y + h * (b41*k1 +
                                   b43*k3));

    k5 = (*f)(x0 + a5*h, *y + h * (b51*k1 +
                                   b53*k3 +
                                   b54*k4));

    k6 = (*f)(x0 + a6*h, *y + h * (b61*k1 +
                                   b64*k4 +
                                   b65*k5));

    k7 = (*f)(x0 + a7*h, *y + h * (b71*k1 +
                                   b74*k4 +
                                   b75*k5 +
                                   b76*k6));

    k8 = (*f)(x0 + a8*h, *y + h * (b81*k1 +
                                   b85*k5 +
                                   b86*k6 +
                                   b87*k7));

    k9 = (*f)(x0 + a9*h, *y + h * (b91*k1 +
                                   b94*k4 +
                                   b95*k5 +
                                   b96*k6 +
                                   b97*k7 +
                                   b98*k8));

    k10 = (*f)(x0 + a10*h, *y + h * (b10_1*k1 +
                                     b10_4*k4 +
                                     b10_5*k5 +
                                     b10_6*k6 +
                                     b10_7*k7 +
                                     b10_8*k8 +
                                     b10_9*k9));

    k11 = (*f)(x0 + h, *y + h * (b11_1*k1 +
                                 b11_4*k4 +
                                 b11_5*k5 +
                                 b11_6*k6 +
                                 b11_7*k7 +
                                 b11_8*k8 +
                                 b11_9*k9 +
                                 b11_10*k10));
                               
    k12 = (*f)(x0, *y + h * (b12_1*k1 +
                             b12_6*k6 +
                             b12_7*k7 +
                             b12_8*k8 +
                             b12_9*k9 +
                             b12_10*k10));
                             
    k13 = (*f)(x0 + h, *y + h * (b13_1*k1 +
                                 b13_4*k4 +
                                 b13_5*k5 +
                                 b13_6*k6 +
                                 b13_7*k7 +
                                 b13_8*k8 +
                                 b13_9*k9 +
                                 b13_10*k10 +
                                 k12));
                                 
    *(y+1) = *y +  h * (c_1_11 * (k1 + k11) +
                                    c6 * k6 +
                          c_7_8 * (k7 + k8) +
                        c_9_10 * (k9 + k10)  );
                        
    return err_factor * (k1 + k11 - k12 - k13);
}




////////////////////////
//   Helper functions //
/////////////////////////////////////////////////////
// _vect_add adds arbitrary numbers of gsl_vectors 
// this is used to calculate the k1...k13 in RKF78
// arguments:
//        o the length of the vectors (int)
//        o the number of vectors (int)
//        o the vectors to be added (gsl_vector)
// returns:
//        o the vector sum of the inputs (gsl_vector)
/////////////////////////////////////////////////////
gsl_vector *_vect_add(int vect_len, int count, ...){
    va_list p;
    int i;
    gsl_vector * sum = gsl_vector_alloc(vect_len);
    gsl_vector * tmp_vec; 
    va_start(p, count);
    
    for (i = 0; i<count; i++){
        tmp_vec = va_arg(p, gsl_vector *);
        gsl_vector_add(sum, tmp_vec);
    }   
    va_end(p);
    return(sum);
}

int _vector_add(int len, double v1[len], double v2[len]){
    for (int i = 0; i < len; i++){
    v1[i] = v1[i] + v2[i]; 
    } 
    return 0;
}

int _populate_history(int hist_len,
                      double history[hist_len][5],
                      int step,
                      double clock,
                      gsl_vector *state){
    history[step][0] = clock;
    for (int i = 1; i < 5; i++){
        history[step][i] = gsl_vector_get(state, i - 1);
    }
    return 0;
}

int _arr2vec(int len, double in_arr[len], gsl_vector *out_vec){
    for (int i = 0; i < len; i++){
        gsl_vector_set(out_vec, i, in_arr[i])
        }
    return 0;
}

int double_pendulum_eom(double t, gsl_vector *in_state, gsl_vector *out_state){
    
    double Th1, Th1_d, Th2, Th2_d;
    Th1 = gsl_vector_get(in_state, 0);
    Th1_d = gsl_vector_get(in_state, 1);
    Th2 = gsl_vector_get(in_state, 2);
    Th2_d = gsl_vector_get(in_state, 3);
    
    double a1, a2, a3, a4, Th1_dd;
    double b1, b2, b3, b4, Th2_dd;
    
    a1 = GRAVITY*(sin(Th2) * cos(Th1 - Th2) - 2*sin(Th1));
    a2 = -(Th2_d*Th2_d + Th1_d*Th1_d*cos(Th1 - Th2));
    a3 = sin(Th1 - Th2);
    a4 = (2-np.cos(Th1 - Th2) * cos(Th1 - Th2));
    Th1_dd = (a1 + (a2 * a3))/a4;

    b1 = 2*GRAVITY*(sin(Th1) * cos(Th1 - Th2) - sin(Th2));
    b2 = 2*Th1_d*Th1_d + Th2_d*Th2_d * cos(Th1 - Th2);
    b3 = sin(Th_1 - Th_2);
    b4 = (2-np.cos(Th1 - Th2)*cos(Th1 - Th2));
    
    Th_dd_2 = (b1 + (b2 * b3))/b4;
    
    gsl_vector_set(out_state, 0, Th1);
    gsl_vector_set(out_state, 0, Th1_d);
    gsl_vector_set(out_state, 0, Th2);
    gsl_vector_set(out_state, 0, Th2_d);
    
    return 0;
}

int main(){
 return 0;
}
