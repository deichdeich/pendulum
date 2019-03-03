#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.

double _rk78(double (*f)(double, double), double *y, double x, double h);
double xfehlberg(double (*f)(double, double),
                   double x,
                   double xmax,
                   double h,
                   double *h_next,
                   double tolerance,
                   double y[],
                   int *iter_count);

double f1(double x, double y);
double f2(double x, double y);
double f3(double x, double y);

int fehlberg( double (*f)(double, double), double y[], double x,
                   double h, double xmax, double *h_next, double tolerance );



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

/*
integrate_rkf78: an embedded Runge-Kutta-Felberg method

Arguments
-------------
    xmin       The starting point along the integration coordinate
    xmax       The ending point
    y0         The inish condish at xmin
    h          The initial step size
    tolerance  The acceptable error in the integration
    history    An array of size (2,N) containing all integration steps, edited in-place
*/


int integrate_rkf78(double (*f)(double, double),
                    double xmin,
                    double xmax,
                    double y0,
                    int hist_len,
                    double history[hist_len][2],
                    double tol,
                    double h){
    
    double stepsize = (xmax - xmin) / hist_len;
    double hpt;
    
    double x1 = xmin;
    double x2 = xmin + stepsize;
    
    double y[2] = {y0, 0.};
    
    for (int i = 0; i < hist_len; i++){
        fehlberg(f, y, x1, h, x2, &hpt, tol);
        history[i][0] = x2;
        history[i][1] = y[1];
        x1 = x2;
        x2 += stepsize;
        y[0] = y[1];
    }
    return 0;

}

int fehlberg(double (*f)(double, double),
                         double y[],
                         double x,
                         double h,
                         double xmax,
                         double *h_next,
                         double tolerance ) {

   static const double err_exponent = 1.0 / 7.0;

   double scale;
   double temp_y[2];
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
   y[1] = y[0];
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

   temp_y[0] = y[0];
   while ( x < xmax ) {
      scale = 1.0;
      for (i = 0; i < ATTEMPTS; i++) {
         
         err = fabs( _rk78(f, temp_y, x, h) );
         
         if (err == 0.0) {
         scale = MAX_SCALE_FACTOR;
         break;
         }
         
         yy = (temp_y[0] == 0.0) ? tolerance : fabs(temp_y[0]);
         
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
      
      temp_y[0] = temp_y[1];         
      
      x += h;
      h *= scale;
      *h_next = h;
      
      if ( last_interval ) break;
      
      if (  x + h > xmax ) { last_interval = 1; h = xmax - x; }
      
      else if ( x + h + 0.5 * h > xmax ) h = 0.5 * h;
   }
   y[1] = temp_y[1];
   return 0;
}



double _rk78(double (*f)(double, double),
             double *y,
             double x0,
             double h){

    /*printf("*y: %f\n", *y);*/
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

    double k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13;
    double h2_7 = a2 * h;

    double thing;
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
    thing = *y +  h * (c_1_11 * (k1 + k11) +
                        c6 * k6 +
                        c_7_8 * (k7 + k8) +
                        c_9_10 * (k9 + k10));
    *(y+1) = thing;
    double thing2 = (k1 + k11 - k12 - k13);
    return err_factor * (k1 + k11 - k12 - k13);
}




double f1(double x, double y){
    return(cos(x));
    }

double f2(double x, double y){
    return(-y);
}

double f3(double x, double y){
    return(cos(x) * pow(sin(x),4) * y);
}

int main(){
return 0;
}

/* Old versions of functions that don't work but that I want to keep for some reason */
/* 
int xintegrate_rkf78(double (*f)(double, double),
                    double start,
                    double end,
                    double h,
                    double *h_next,
                    double tolerance,
                    int N,
                    double init_state[2],
                    double history[2][N]){
    int dummy = 0;
    double x = start;
    double x_max = 1.0;
    history[0][0] = init_state[0];
    history[1][0] = init_state[1];
    
    for(int i = 1; i < N; i++){
        double current_state[2];
        current_state[0] = history[0][i-1];
        current_state[1] = history[1][i-1];
        
        h = feldberg(f,
                      x,
                      x_max,
                      h,
                      h_next,
                      tolerance,
                      current_state, &dummy);
                      
        if(current_state[0] > end){
            break;
            }
        
        history[0][i] = current_state[0];
        history[1][i] = current_state[1];
    }
    return 0;
}


Unlike RK4, this is an adaptive method, so specifying the number of steps doesn't
make as much sense.  Instead, you specify the start and end of the integration with
x_0 and x_max.  Therefore, integrate_rkf78 is called *for a single step* by another
function (yet to be written) that calls it in order to fill out a whole history array.

Python will then look at the output and determine if more steps are required.


double xfehlberg(double (*f)(double, double),
                    double x,
                    double xmax,
                    double h,
                    double *h_next,
                    double tolerance,
                    double y[],
                    int *iter_count){
    
    const double err_exponent = 1. / 7.;
    
    double scale;
    double temp_y[2];
    double err;
    double yy;
    int i;
    int last_interval = 0;

    
    Check for stupid things:  is h positive?  Is it integrating forward?  if not,
    exit with error code -2
    
    if (h <= 0.0) return -2;


    *h_next = h;
    y[1] = y[0];
    temp_y[0] = y[0];  
    
    while (x < xmax){
        scale = 1.0;

        for (i = 0; i < ATTEMPTS; i++){
         *iter_count += 1;
     
         err = fabs(_rk78(f, temp_y, x, h));
         printf("err: %f\n", err);
         if (err == 0.0) {
            printf("test 1\n");
            scale = MAX_SCALE_FACTOR;
            break;
            }
         printf("test 2\n");
     
         yy = (temp_y[0] == 0.0) ? tolerance : fabs(temp_y[0]);
     
         scale = 0.8 * pow(tolerance * yy /  err, err_exponent);
         scale = min(max(scale, MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);
         printf("scale: %f\n", scale);
         if (err < (tolerance * yy)){
            printf("test 3\n");
            break;
            }

         h *= scale;
        if ( last_interval ) break;
        if (  x + h > xmax ) {
            last_interval = 1;
            h = xmax - x;
            }

         else if ((x + h + 0.5 * h) > xmax){
            printf("test 5\n");
            h = 0.5 * h;
            }
        }
        if (i >= ATTEMPTS){
            printf("test 6\n");
            *h_next = h * scale;
            return -1;
        }

        temp_y[0] = temp_y[1];         
        x += h;
        h *= scale;
        *h_next = h;
        printf("h: %f\n", h);
        printf("x: %f\n", x);

        }
    y[1] = temp_y[1];
    return h;
}
*/

