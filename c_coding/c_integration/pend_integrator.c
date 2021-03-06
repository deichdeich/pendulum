#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>


#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.

#define GRAVITY 9.8

int rkf78(double (*f)(double, gsl_vector *, gsl_vector *),
          double xmin,
          double xmax,
          double init_state[4],
          int hist_len,
          double history[hist_len][5],
          double tol,
          double h);

int _stepper(double (*f)(double, gsl_vector *, gsl_vector *),
             gsl_vector *in_state_vec,
             gsl_vector *out_state_vec,
             double x,
             double h,
             double xmax,
             double *h_next,
             double tolerance );
             
double _single_step(double (*f)(double, gsl_vector *, gsl_vector *),
                    gsl_vector *in_vec,
                    gsl_vector *out_vec,
                    double x0,
                    double h);

int *_vect_add(gsl_vector * sum, int vect_len, int count, ...);
int _populate_history(int hist_len,
                      double history[hist_len][5],
                      int step,
                      double clock,
                      gsl_vector *state);

int _arr2vec(int len, double in_arr[len], gsl_vector *out_vec);

static struct {
    gsl_vector *k1_in;
    gsl_vector *k2_in;
    gsl_vector *k3_in;
    gsl_vector *k4_in;
    gsl_vector *k5_in;
    gsl_vector *k6_in;
    gsl_vector *k7_in;
    gsl_vector *k8_in;
    gsl_vector *k9_in;
    gsl_vector *k10_in;
    gsl_vector *k11_in;
    gsl_vector *k12_in;
    gsl_vector *k13_in;
    gsl_vector *k1_out;
    gsl_vector *k2_out;
    gsl_vector *k3_out;
    gsl_vector *k4_out;
    gsl_vector *k5_out;
    gsl_vector *k6_out;
    gsl_vector *k7_out;
    gsl_vector *k8_out;
    gsl_vector *k9_out;
    gsl_vector *k10_out;
    gsl_vector *k11_out;
    gsl_vector *k12_out;
    gsl_vector *k13_out;
    
    gsl_vector *k3k1_out;
    gsl_vector *k3k2_out;
    
    gsl_vector *k4k1_out;
    gsl_vector *k4k3_out;
    
    gsl_vector *k5k1_out;
    gsl_vector *k5k3_out;
    gsl_vector *k5k4_out;
    
    gsl_vector *k6k1_out;
    gsl_vector *k6k4_out;
    gsl_vector *k6k5_out;

    gsl_vector *k7k1_out;
    gsl_vector *k7k4_out;
    gsl_vector *k7k5_out;
    gsl_vector *k7k6_out;

    gsl_vector *k8k1_out;
    gsl_vector *k8k5_out;
    gsl_vector *k8k6_out;
    gsl_vector *k8k7_out;

    gsl_vector *k9k1_out;
    gsl_vector *k9k4_out;
    gsl_vector *k9k5_out;
    gsl_vector *k9k6_out;
    gsl_vector *k9k7_out;
    gsl_vector *k9k8_out;

    gsl_vector *k10k1_out;
    gsl_vector *k10k4_out;
    gsl_vector *k10k5_out;
    gsl_vector *k10k6_out;
    gsl_vector *k10k7_out;
    gsl_vector *k10k8_out;
    gsl_vector *k10k9_out;

    gsl_vector *k11k1_out;
    gsl_vector *k11k4_out;
    gsl_vector *k11k5_out;
    gsl_vector *k11k6_out;
    gsl_vector *k11k7_out;
    gsl_vector *k11k8_out;
    gsl_vector *k11k9_out;
    gsl_vector *k11k10_out;

    gsl_vector *k12k1_out;
    gsl_vector *k12k6_out;
    gsl_vector *k12k7_out;
    gsl_vector *k12k8_out;
    gsl_vector *k12k9_out;
    gsl_vector *k12k10_out;    

    gsl_vector *k13k1_out;
    gsl_vector *k13k4_out;
    gsl_vector *k13k5_out;
    gsl_vector *k13k6_out;
    gsl_vector *k13k7_out;
    gsl_vector *k13k8_out;
    gsl_vector *k13k9_out;
    gsl_vector *k13k10_out;
    gsl_vector *k13k12_out;

    gsl_vector *c_1_11_vec;
    gsl_vector *c_6_vec;
    gsl_vector *c_7_8_vec;
    gsl_vector *c_9_10_vec;
    gsl_vector *c_tot_vec;

    gsl_vector *err_vec;
    gsl_vector *ek12;
    gsl_vector *ek13;
    
    gsl_vector *temp_in_vec;
    gsl_vector *temp_out_vec;
    
} integration_vectors;

static int init_count = 0;
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
void integration_vector_init(){
    init_count+=1;
    int i, Nelt;
    gsl_vector **p;

    Nelt = sizeof(integration_vectors) / sizeof(integration_vectors.k1_in);    // number of elements in x.

    p = &integration_vectors.k1_in;                          // p is pointer to x.a1, hence
                                           // p[0] points to same as x.a1;
                                           // p[1] points to same as x.a2; etc.
    for (i=0 ; i < Nelt ; i++, p++) {
        *p = gsl_vector_calloc(4);
        }
}


void integration_vector_free(){
    int i, Nelt;
    gsl_vector **p;

    Nelt = sizeof(integration_vectors) / sizeof(integration_vectors.k1_in);    // number of elements in x.

    p = &integration_vectors.k1_in;                          // p is pointer to x.a1, hence
                                           // p[0] points to same as x.a1;
                                           // p[1] points to same as x.a2; etc.
    for (i=0 ; i < Nelt ; i++, p++) {
        gsl_vector_free(*p);
        }
}

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
void print_vec(gsl_vector * vec){
    for (int i = 0; i < 4; i++){
        //printf("%f, ", gsl_vector_get(vec, i));
    }
    //printf("\n");
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
int rkf78(double (*f)(double, gsl_vector *, gsl_vector *),
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


    gsl_vector *init_state_vec = gsl_vector_calloc(4);
    _arr2vec(4, init_state, init_state_vec);
            //printf("0 in: ");
        //print_vec(init_state_vec);

    gsl_vector *out_state_vec = gsl_vector_calloc(4);
            //printf("0 out: ");
        //print_vec(out_state_vec);
    for (int i = 0; i < hist_len; i++){
        //printf("1 in: ");
        //print_vec(init_state_vec);
        //printf("1 out: ");
        //print_vec(out_state_vec);
        _stepper(f, init_state_vec, out_state_vec, x1, h, x2, &hpt, tol);
        //printf("2 in: ");
        //print_vec(init_state_vec);
        //printf("2 out: ");
        //print_vec(out_state_vec);
        _populate_history(hist_len, history, i, x2, out_state_vec);
        x1 = x2;
        x2 += stepsize;
        gsl_vector_memcpy(init_state_vec, out_state_vec);

    }
    ////printf("%d\n", init_count);    

    return 0;
}

int _stepper(double (*f)(double, gsl_vector *, gsl_vector *),
             gsl_vector *in_state_vec,
             gsl_vector *out_state_vec,
             double x,
             double h,
             double xmax,
             double *h_next,
             double tolerance ){

  
    integration_vector_init();
   static const double err_exponent = 1.0 / 7.0;
    
   double scale;
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
        //printf("3 in: ");
        //print_vec(in_state_vec);
        //printf("3 out: ");
        //print_vec(out_state_vec);
   gsl_vector_memcpy(out_state_vec, in_state_vec);
        //printf("4 in: ");
        //print_vec(in_state_vec);
        //printf("4 out: ");
        //print_vec(out_state_vec);
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

   gsl_vector_memcpy(integration_vectors.temp_in_vec, in_state_vec);

   while ( x < xmax ) {
      scale = 1.0;
      for (i = 0; i < ATTEMPTS; i++) {
        //printf("5 in: ");
        //print_vec(integration_vectors.temp_in_vec);
        //printf("5 out: ");
        //print_vec(integration_vectors.temp_out_vec);
         err = fabs(_single_step(f,
                                 integration_vectors.temp_in_vec,
                                 integration_vectors.temp_out_vec,
                                 x,
                                 h));
        //printf("6 in: ");
        //print_vec(integration_vectors.temp_in_vec);
        //printf("6 out: ");
        //print_vec(integration_vectors.temp_out_vec);
         if (err == 0.0) {

         scale = MAX_SCALE_FACTOR;
         break;
         }

         double statenorm = gsl_blas_dnrm2(integration_vectors.temp_in_vec);
         
         yy = (statenorm == 0.0) ? tolerance : statenorm;
         
         

         
         scale = 0.8 * pow( tolerance * yy /  err , err_exponent );
         scale = min( max(scale,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

         if ( err < ( tolerance * yy ) ) break;

         h *= scale;

         if ( x + h > xmax ) h = xmax - x;

         else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;

      
      if ( i >= ATTEMPTS ) {
            *h_next = h * scale;
            return -1;
            }
    }
        //printf("temp_in_vec: ");
        //print_vec(integration_vectors.temp_in_vec);
      gsl_vector_memcpy(integration_vectors.temp_in_vec,
                        integration_vectors.temp_out_vec);
      x += h;
      h *= scale;
      *h_next = h;

      if ( last_interval ) {

      break;
      }

      if (  x + h > xmax ) { last_interval = 1; h = xmax - x; }
      
      else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
   }
   gsl_vector_memcpy(out_state_vec, integration_vectors.temp_out_vec);
   integration_vector_free();
   //printf("out_state_vec:");
    //print_vec(out_state_vec);
   return 0;
}



double _single_step(double (*f)(double, gsl_vector *, gsl_vector *),
                    gsl_vector *in_vec,
                    gsl_vector *out_vec,
                    double x0,
                    double h){
        //printf("7 in: ");
        //print_vec(in_vec);
        //printf("7 out: ");
        //print_vec(out_vec);
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
    double h2_7 = a2 * h;
    
    //////////////
    // k1
    /////////////
    gsl_vector_memcpy(integration_vectors.k1_in, in_vec);
    (*f)(x0, integration_vectors.k1_in, integration_vectors.k1_out);

    //////////////
    // k2
    ///////////// 
    gsl_vector_memcpy(integration_vectors.k2_in,
                      integration_vectors.k1_out);
    gsl_blas_dscal(h2_7, integration_vectors.k2_in);
    gsl_vector_add(integration_vectors.k2_in, in_vec);
    
    (*f)(x0 + h2_7, integration_vectors.k2_in,
                    integration_vectors.k2_out);
    //printf("k2: ");
    //print_vec(integration_vectors.k2_out);

    //////////////
    // k3
    /////////////
    gsl_vector_memcpy(integration_vectors.k3k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k3k2_out, integration_vectors.k2_out);
    gsl_blas_dscal(b31, integration_vectors.k3k1_out);
    gsl_blas_dscal(b32, integration_vectors.k3k2_out);
    _vect_add(integration_vectors.k3_in, 4, 2, integration_vectors.k3k1_out,
                                               integration_vectors.k3k2_out);
    gsl_blas_dscal(h, integration_vectors.k3_in);
    gsl_vector_add(integration_vectors.k3_in, in_vec);
    
    (*f)(x0 + a3*h, integration_vectors.k3_in, integration_vectors.k3_out);
    //printf("k3: ");
    //print_vec(integration_vectors.k3_out);    //////////////
    // k4
    /////////////
    gsl_vector_memcpy(integration_vectors.k4k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k4k3_out, integration_vectors.k3_out);
    gsl_blas_dscal(b41, integration_vectors.k4k1_out);
    gsl_blas_dscal(b43, integration_vectors.k4k3_out);
    _vect_add(integration_vectors.k4_in, 4, 2, integration_vectors.k4k1_out,
                                               integration_vectors.k4k3_out);
    gsl_blas_dscal(h, integration_vectors.k4_in);
    gsl_vector_add(integration_vectors.k4_in, in_vec);
    
    (*f)(x0 + a4*h, integration_vectors.k4_in, integration_vectors.k4_out);
    //printf("k4: ");
    //print_vec(integration_vectors.k4_out);    //////////////
    // k5
    /////////////
    gsl_vector_memcpy(integration_vectors.k5k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k5k3_out, integration_vectors.k3_out);
    gsl_vector_memcpy(integration_vectors.k5k4_out, integration_vectors.k4_out);
    gsl_blas_dscal(b51, integration_vectors.k5k1_out);
    gsl_blas_dscal(b53, integration_vectors.k5k3_out);
    gsl_blas_dscal(b54, integration_vectors.k5k4_out);
    _vect_add(integration_vectors.k5_in, 4, 3, integration_vectors.k5k1_out,
                                               integration_vectors.k5k3_out,
                                               integration_vectors.k5k4_out);
    gsl_blas_dscal(h, integration_vectors.k5_in);
    gsl_vector_add(integration_vectors.k5_in, in_vec);    
    
    (*f)(x0 + a5*h, integration_vectors.k5_in, integration_vectors.k5_out);
    //printf("k5: ");
    //print_vec(integration_vectors.k5_out);
    // k6
    /////////////
    gsl_vector_memcpy(integration_vectors.k6k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k6k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k6k5_out, integration_vectors.k5_out);
    gsl_blas_dscal(b61, integration_vectors.k6k1_out);
    gsl_blas_dscal(b64, integration_vectors.k6k4_out);
    gsl_blas_dscal(b65, integration_vectors.k6k5_out);
    _vect_add(integration_vectors.k6_in, 4, 3, integration_vectors.k6k1_out,
                                               integration_vectors.k6k4_out,
                                               integration_vectors.k6k5_out);
    gsl_blas_dscal(h, integration_vectors.k6_in);
    gsl_vector_add(integration_vectors.k6_in, in_vec);    
    
    (*f)(x0 + a6*h, integration_vectors.k6_in, integration_vectors.k6_out);
    //printf("k6: ");
    //print_vec(integration_vectors.k6_out);
    //////////////
    // k7
    /////////////
    gsl_vector_memcpy(integration_vectors.k7k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k7k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k7k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k7k6_out, integration_vectors.k6_out);
    gsl_blas_dscal(b71, integration_vectors.k7k1_out);
    gsl_blas_dscal(b74, integration_vectors.k7k4_out);
    gsl_blas_dscal(b75, integration_vectors.k7k5_out);
    gsl_blas_dscal(b76, integration_vectors.k7k6_out);
    _vect_add(integration_vectors.k7_in, 4, 4, integration_vectors.k7k1_out,
                                               integration_vectors.k7k4_out,
                                               integration_vectors.k7k5_out,
                                               integration_vectors.k7k6_out);
    gsl_blas_dscal(h, integration_vectors.k7_in);
    gsl_vector_add(integration_vectors.k7_in, in_vec);    
    
    (*f)(x0 + a7*h, integration_vectors.k7_in, integration_vectors.k7_out);
    //printf("k7: ");
    //print_vec(integration_vectors.k7_out);
    //////////////
    // k8
    /////////////
    gsl_vector_memcpy(integration_vectors.k8k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k8k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k8k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k8k7_out, integration_vectors.k7_out);
    gsl_blas_dscal(b81, integration_vectors.k8k1_out);
    gsl_blas_dscal(b85, integration_vectors.k8k5_out);
    gsl_blas_dscal(b86, integration_vectors.k8k6_out);
    gsl_blas_dscal(b87, integration_vectors.k8k7_out);
    _vect_add(integration_vectors.k8_in, 4, 4, integration_vectors.k8k1_out,
                                               integration_vectors.k8k5_out,
                                               integration_vectors.k8k6_out,
                                               integration_vectors.k8k7_out);
    gsl_blas_dscal(h, integration_vectors.k8_in);
    gsl_vector_add(integration_vectors.k8_in, in_vec);    
    
    (*f)(x0 + a8*h, integration_vectors.k8_in, integration_vectors.k8_out);
   
    //printf("k8: ");
    //print_vec(integration_vectors.k8_out);   
    //////////////
    // k9
    /////////////
    gsl_vector_memcpy(integration_vectors.k9k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k9k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k9k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k9k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k9k7_out, integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k9k8_out, integration_vectors.k8_out);
    gsl_blas_dscal(b91, integration_vectors.k9k1_out);
    gsl_blas_dscal(b94, integration_vectors.k9k4_out);
    gsl_blas_dscal(b95, integration_vectors.k9k5_out);
    gsl_blas_dscal(b96, integration_vectors.k9k6_out);
    gsl_blas_dscal(b97, integration_vectors.k9k7_out);
    gsl_blas_dscal(b98, integration_vectors.k9k8_out);
    _vect_add(integration_vectors.k9_in, 4, 6, integration_vectors.k9k1_out,
                                               integration_vectors.k9k4_out,
                                               integration_vectors.k9k5_out,
                                               integration_vectors.k9k6_out,
                                               integration_vectors.k9k7_out,
                                               integration_vectors.k9k8_out);
                                               
    gsl_blas_dscal(h, integration_vectors.k9_in);
    gsl_vector_add(integration_vectors.k9_in, in_vec);  
    (*f)(x0 + a9*h, integration_vectors.k9_in, integration_vectors.k9_out);
    //printf("k9: ");
    //print_vec(integration_vectors.k9_out);
    //////////////
    // k10
    /////////////
    gsl_vector_memcpy(integration_vectors.k10k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k10k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k10k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k10k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k10k7_out, integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k10k8_out, integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k10k9_out, integration_vectors.k9_out);
    gsl_blas_dscal(b10_1, integration_vectors.k10k1_out);
    gsl_blas_dscal(b10_4, integration_vectors.k10k4_out);
    gsl_blas_dscal(b10_5, integration_vectors.k10k5_out);
    gsl_blas_dscal(b10_6, integration_vectors.k10k6_out);
    gsl_blas_dscal(b10_7, integration_vectors.k10k7_out);
    gsl_blas_dscal(b10_8, integration_vectors.k10k8_out);
    gsl_blas_dscal(b10_9, integration_vectors.k10k9_out);
    _vect_add(integration_vectors.k10_in, 4, 7, integration_vectors.k10k1_out,
                                                integration_vectors.k10k4_out,
                                                integration_vectors.k10k5_out,
                                                integration_vectors.k10k6_out,
                                                integration_vectors.k10k7_out,
                                                integration_vectors.k10k8_out,
                                                integration_vectors.k10k9_out);
    gsl_blas_dscal(h, integration_vectors.k10_in);
    gsl_vector_add(integration_vectors.k10_in, in_vec);  
    (*f)(x0 + a10*h, integration_vectors.k10_in, integration_vectors.k10_out);
    //printf("k10: ");
    //print_vec(integration_vectors.k10_out);
    //////////////
    // k11
    /////////////
    gsl_vector_memcpy(integration_vectors.k11k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k11k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k11k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k11k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k11k7_out, integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k11k8_out, integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k11k9_out, integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k11k10_out, integration_vectors.k10_out);
    gsl_blas_dscal(b11_1, integration_vectors.k11k1_out);
    gsl_blas_dscal(b11_4, integration_vectors.k11k4_out);
    gsl_blas_dscal(b11_5, integration_vectors.k11k5_out);
    gsl_blas_dscal(b11_6, integration_vectors.k11k6_out);
    gsl_blas_dscal(b11_7, integration_vectors.k11k7_out);
    gsl_blas_dscal(b11_8, integration_vectors.k11k8_out);
    gsl_blas_dscal(b11_9, integration_vectors.k11k9_out);
    gsl_blas_dscal(b11_10, integration_vectors.k11k10_out);
    _vect_add(integration_vectors.k11_in, 4, 8, integration_vectors.k11k1_out,
                                                integration_vectors.k11k4_out,
                                                integration_vectors.k11k5_out,
                                                integration_vectors.k11k6_out,
                                                integration_vectors.k11k7_out,
                                                integration_vectors.k11k8_out,
                                                integration_vectors.k11k9_out,
                                                integration_vectors.k11k10_out);
    gsl_blas_dscal(h, integration_vectors.k11_in);
    gsl_vector_add(integration_vectors.k11_in, in_vec);  
    (*f)(x0 + h, integration_vectors.k11_in, integration_vectors.k11_out);
    //printf("k11: ");
    //print_vec(integration_vectors.k11_out);
    //////////////
    // k12
    /////////////
    gsl_vector_memcpy(integration_vectors.k12k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k12k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k12k7_out, integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k12k8_out, integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k12k9_out, integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k12k10_out, integration_vectors.k10_out);
    gsl_blas_dscal(b12_1, integration_vectors.k12k1_out);
    gsl_blas_dscal(b12_6, integration_vectors.k12k6_out);
    gsl_blas_dscal(b12_7, integration_vectors.k12k7_out);
    gsl_blas_dscal(b12_8, integration_vectors.k12k8_out);
    gsl_blas_dscal(b12_9, integration_vectors.k12k9_out);
    gsl_blas_dscal(b12_10, integration_vectors.k12k10_out);
    _vect_add(integration_vectors.k12_in, 4, 6, integration_vectors.k12k1_out,
                                                integration_vectors.k12k6_out,
                                                integration_vectors.k12k7_out,
                                                integration_vectors.k12k8_out,
                                                integration_vectors.k12k9_out,
                                                integration_vectors.k12k10_out);
    gsl_blas_dscal(h, integration_vectors.k12_in);
    gsl_vector_add(integration_vectors.k12_in, in_vec); 
                           
    (*f)(x0, integration_vectors.k12_in, integration_vectors.k12_out);
    //printf("k12: ");
    //print_vec(integration_vectors.k12_out);
    //////////////
    // k13
    /////////////
    gsl_vector_memcpy(integration_vectors.k13k1_out, integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k13k4_out, integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k13k5_out, integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k13k6_out, integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k13k7_out, integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k13k8_out, integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k13k9_out, integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k13k10_out, integration_vectors.k10_out);
    gsl_vector_memcpy(integration_vectors.k13k12_out, integration_vectors.k12_out);
    gsl_blas_dscal(b13_1, integration_vectors.k13k1_out);
    gsl_blas_dscal(b13_4, integration_vectors.k13k4_out);
    gsl_blas_dscal(b13_5, integration_vectors.k13k5_out);
    gsl_blas_dscal(b13_6, integration_vectors.k13k6_out);
    gsl_blas_dscal(b13_7, integration_vectors.k13k7_out);
    gsl_blas_dscal(b13_8, integration_vectors.k13k8_out);
    gsl_blas_dscal(b13_9, integration_vectors.k13k9_out);
    gsl_blas_dscal(b13_10, integration_vectors.k13k10_out);
    _vect_add(integration_vectors.k13_in, 4, 9, integration_vectors.k13k1_out,
                                                integration_vectors.k13k4_out,
                                                integration_vectors.k13k5_out,
                                                integration_vectors.k13k6_out,
                                                integration_vectors.k13k7_out,
                                                integration_vectors.k13k8_out,
                                                integration_vectors.k13k9_out,
                                                integration_vectors.k13k10_out,
                                                integration_vectors.k13k12_out);
    gsl_blas_dscal(h, integration_vectors.k13_in);
    gsl_vector_add(integration_vectors.k13_in, in_vec); 
                           
    (*f)(x0, integration_vectors.k13_in, integration_vectors.k13_out);                             
    //printf("k13: ");
    //print_vec(integration_vectors.k13_out);    
    //////////////
    // out_vec
    /////////////
    _vect_add(integration_vectors.c_1_11_vec, 4, 2, integration_vectors.k1_out,
                                                    integration_vectors.k11_out);
    gsl_vector_memcpy(integration_vectors.c_6_vec, integration_vectors.k6_out);
    _vect_add(integration_vectors.c_7_8_vec, 4, 2, integration_vectors.k7_out,
                                                   integration_vectors.k8_out);
    _vect_add(integration_vectors.c_9_10_vec, 4, 2, integration_vectors.k9_out,
                                                    integration_vectors.k10_out);
    gsl_blas_dscal(c_1_11, integration_vectors.c_1_11_vec);
    gsl_blas_dscal(c6, integration_vectors.c_6_vec);
    gsl_blas_dscal(c_7_8, integration_vectors.c_7_8_vec);
    gsl_blas_dscal(c_9_10, integration_vectors.c_9_10_vec);
    _vect_add(integration_vectors.c_tot_vec, 4, 4, integration_vectors.c_1_11_vec,
                                                   integration_vectors.c_6_vec,
                                                   integration_vectors.c_7_8_vec,
                                                   integration_vectors.c_9_10_vec);
    
    gsl_blas_dscal(h, integration_vectors.c_tot_vec);
    gsl_vector_add(integration_vectors.c_tot_vec, in_vec);
    gsl_vector_memcpy(out_vec, integration_vectors.c_tot_vec);
    //printf("single_step out_vec: ");
    //print_vec(out_vec);
    //////////////
    // err_factor
    /////////////
    gsl_vector_memcpy(integration_vectors.ek12, integration_vectors.k12_out);
    gsl_vector_memcpy(integration_vectors.ek13, integration_vectors.k13_out);
    gsl_blas_dscal(-1, integration_vectors.ek12);
    gsl_blas_dscal(-1, integration_vectors.ek13);
    
    _vect_add(integration_vectors.err_vec, 4, 4, integration_vectors.k1_out,
                                                 integration_vectors.k11_out,
                                                 integration_vectors.ek12,
                                                 integration_vectors.ek13);   
    
    
    
    
    return err_factor * gsl_blas_dnrm2(integration_vectors.err_vec);
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
    a4 = (2-cos(Th1 - Th2) * cos(Th1 - Th2));
    Th1_dd = (a1 + (a2 * a3))/a4;

    b1 = 2*GRAVITY*(sin(Th1) * cos(Th1 - Th2) - sin(Th2));
    b2 = 2*Th1_d*Th1_d + Th2_d*Th2_d * cos(Th1 - Th2);
    b3 = sin(Th1 - Th2);
    b4 = (2-cos(Th1 - Th2)*cos(Th1 - Th2));
    
    Th2_dd = (b1 + (b2 * b3))/b4;
    
    gsl_vector_set(out_state, 0, Th1_d);
    gsl_vector_set(out_state, 1, Th1_dd);
    gsl_vector_set(out_state, 2, Th2_d);
    gsl_vector_set(out_state, 3, Th2_dd);
    
    return 0;
}


////////////////////////
//   Helper functions 
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
int *_vect_add(gsl_vector * sum, int vect_len, int count, ...){
    va_list p;
    int i;
    gsl_vector *tmp_vec; 
    va_start(p, count);
    
    for (i = 0; i<count; i++){
        tmp_vec = va_arg(p, gsl_vector *);
        gsl_blas_daxpy(1, tmp_vec, sum);
    }   
    va_end(p);
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
        gsl_vector_set(out_vec, i, in_arr[i]);
        }
    return 0;
}

int main(){
 return 0;
}
