/*
ad_rkf78.c:
Alex Deich's implementation of an RKF7(8) integrator.
Date: March, 2019
alexanderdeich@montana.edu
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

/* Why doesn't C have built-in max, min functions? */
#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.

#define GRAVITY 9.8

/* declaring all the helper functions, defined below the heavy-lifting integration code */
static int stepper(double (*f) (double, gsl_vector *, gsl_vector *),
	     gsl_vector * in_state_vec, gsl_vector * out_state_vec,
	     double x, double h, double xmax, double *h_next,
	     double tolerance);
static double single_step(double (*f) (double, gsl_vector *, gsl_vector *),
		    gsl_vector * in_vec, gsl_vector * out_vec, double x0,
		    double h);
static int *vect_add(gsl_vector * sum, int vect_len, int count, ...);
static int populate_history(int hist_len, double history[hist_len][5], int step,
		      double clock, gsl_vector * state);
static int arr2vec(int len, double in_arr[len], gsl_vector * out_vec);
static void integration_vector_init();
static void integration_vector_free();
static void print_vec(gsl_vector * vec);

/* declaring the integration vector struct*/
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

////////////////////////////////////////////////////////////////////
//  rkf78: an embedded Runge-Kutta-Felberg method
//
//  It's an adaptive time-step method.  My implementation will take a grid of timesteps
// which the integrator will adaptively step through to achieve the desired tolerance
// at each timestep.  So, even though the time is initially given in a rigid grid (and
// ultimately returned in the same rigid grid) getting between those grids takes a variable
// number of timesteps to ensure the desired precision.
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
int rkf78(double (*f) (double, gsl_vector *, gsl_vector *),
	  double xmin,
	  double xmax,
	  double init_state[4],
	  int hist_len, double history[hist_len][5], double tol, double h)
{

    double stepsize = (xmax - xmin) / hist_len;
    double hpt;

    double x1 = xmin;
    double x2 = xmin + stepsize;

    /* initialize initial state */
    gsl_vector *init_state_vec = gsl_vector_calloc(4);
    /* load the array from python into the gsl_vector */
    arr2vec(4, init_state, init_state_vec);

    /* initialize the final state */
    gsl_vector *out_state_vec = gsl_vector_calloc(4);

    /* step through all the times */
    for (int i = 0; i < hist_len; i++) {
	stepper(f, init_state_vec, out_state_vec, x1, h, x2, &hpt, tol);
	populate_history(hist_len, history, i, x2, out_state_vec);
	x1 = x2;
	x2 += stepsize;
	gsl_vector_memcpy(init_state_vec, out_state_vec);

    }

    return 0;
}

/////////////////////
// stepper steps the state from one timestamp to another.  It makes sure that the error
// in the final state is less than a given tolerance.
//
// Arguments:
//      *f: pointer to the function which gives the derivative of a system at a given time
//          and given initial condition.  It's an equation of motion.
//      in_state_vec: pointer to the initial condition, gsl_vector
//      out_state_vec: pointer to the state, to be filled, at the time x0+h, gsl_vector
//      x: the clock value corresponding to in_vec, double
//      h: the initial timestep guess, double
//      xmax: the final clock value to integrate to, double
//      *h_next: pointer to where the next timestep is stored, double
//      tolerance:  The error in the integration is guaranteed to be less than this, double
//
//  Returns:
//      0 if successful
//      No failure case yet
//
/////////////////////
static int stepper(double (*f) (double, gsl_vector *, gsl_vector *),
	     gsl_vector * in_state_vec,
	     gsl_vector * out_state_vec,
	     double x,
	     double h, double xmax, double *h_next, double tolerance)
{


    integration_vector_init();
    static const double err_exponent = 1.0 / 7.0;

    double scale;
    double err;
    double yy;
    int i;
    int last_interval = 0;

    // Verify that the step size is positive and that the upper endpoint //
    // of integration is greater than the initial enpoint.               //

    if (xmax < x || h <= 0.0)
	return -2;

    // If the upper endpoint of the independent variable agrees with the //
    // initial value of the independent variable.  Set the value of the  //
    // dependent variable and return success.                            //

    *h_next = h;
    gsl_vector_memcpy(out_state_vec, in_state_vec);
    if (xmax == x)
	return 0;

    // ensure that the step size h is not larger than the length of the //
    // integration interval.                                            //

    if (h > (xmax - x)) {
	h = xmax - x;
	last_interval = 1;
    }
    // Redefine the error tolerance to an error tolerance per unit    //
    // length of the integration interval.                            //

    tolerance /= (xmax - x);

    // Integrate the diff eq y'=f(x,y) from x=x to x=xmax trying to  //
    // maintain an error less than tolerance * (xmax-x) using an     //
    // initial step size of h and initial value: y = y[0]            //

    gsl_vector_memcpy(integration_vectors.temp_in_vec, in_state_vec);

    while (x < xmax) {
	scale = 1.0;
	for (i = 0; i < ATTEMPTS; i++) {
	    err = fabs(single_step(f,
				    integration_vectors.temp_in_vec,
				    integration_vectors.temp_out_vec,
				    x, h));
	    if (err == 0.0) {

		scale = MAX_SCALE_FACTOR;
		break;
	    }

	    double statenorm =
		gsl_blas_dnrm2(integration_vectors.temp_in_vec);

	    yy = (statenorm == 0.0) ? tolerance : statenorm;




	    scale = 0.8 * pow(tolerance * yy / err, err_exponent);
	    scale = min(max(scale, MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

	    if (err < (tolerance * yy))
		break;

	    h *= scale;

	    if (x + h > xmax)
		h = xmax - x;

	    else if (x + h + 0.5 * h > xmax)
		h = 0.5 * h;


	    if (i >= ATTEMPTS) {
		*h_next = h * scale;
		return -1;
	    }
	}

	gsl_vector_memcpy(integration_vectors.temp_in_vec,
			  integration_vectors.temp_out_vec);
	x += h;
	h *= scale;
	*h_next = h;

	if (last_interval) {

	    break;
	}

	if (x + h > xmax) {
	    last_interval = 1;
	    h = xmax - x;
	}

	else if (x + h + 0.5 * h > xmax)
	    h = 0.5 * h;
    }
    gsl_vector_memcpy(out_state_vec, integration_vectors.temp_out_vec);
    integration_vector_free();
    return 0;
}


////////////////
// single_step computes the next value of the system for a single timestep.
// It's a shitty, awful function that just encodes the RKF7(8) Butcher tableau.
// I hate editing this function.  Don't touch it if you don't think you have to.
// It basically makes sense if you look at any pseudocode implementation of RKF7(8); not sure which source I used. 
//
// It relies on the integration_vectors struct to hold each interstitial integration vector.
// 
// Arguments:
//      *f: pointer to the function which gives the derivative of a system at a given time
//          and given initial condition.  It's an equation of motion.
//      in_vec: pointer to the initial condition, gsl_vector
//      out_vec: pointer to the state, to be filled, at the time x0+h, gsl_vector
//      x0: the clock value corresponding to in_vec
//      h: the timestep
//
// Returns:
//      err: The error in the 7th-order timestep as evaluated by the 8th-order. This is used
//           by stepper() to _determine the size of the next timestep (if err is too big).  Double.
//
////////////////
static double single_step(double (*f) (double, gsl_vector *, gsl_vector *),
		    gsl_vector * in_vec,
		    gsl_vector * out_vec, double x0, double h)
{
    const double c_1_11 = 41.0 / 840.0;
    const double c6 = 34.0 / 105.0;
    const double c_7_8 = 9.0 / 35.0;
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
    const double b74 = 125.0 / 108.0;
    const double b75 = -260.0 / 108.0;
    const double b76 = 250.0 / 108.0;
    const double b81 = 31.0 / 300.0;
    const double b85 = 61.0 / 225.0;
    const double b86 = -2.0 / 9.0;
    const double b87 = 13.0 / 900.0;
    const double b91 = 2.0;
    const double b94 = -53.0 / 6.0;
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
    const double b12_6 = -6.0 / 41.0;
    const double b12_7 = -3.0 / 205.0;
    const double b12_8 = -3.0 / 41.0;
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

    const double err_factor = -41.0 / 840.0;
    double h2_7 = a2 * h;

    //////////////
    // k1
    /////////////
    gsl_vector_memcpy(integration_vectors.k1_in, in_vec);
    (*f) (x0, integration_vectors.k1_in, integration_vectors.k1_out);

    //////////////
    // k2
    ///////////// 
    gsl_vector_memcpy(integration_vectors.k2_in,
		      integration_vectors.k1_out);
    gsl_blas_dscal(h2_7, integration_vectors.k2_in);
    gsl_vector_add(integration_vectors.k2_in, in_vec);

    (*f) (x0 + h2_7, integration_vectors.k2_in,
	  integration_vectors.k2_out);

    //////////////
    // k3
    /////////////
    gsl_vector_memcpy(integration_vectors.k3k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k3k2_out,
		      integration_vectors.k2_out);
    gsl_blas_dscal(b31, integration_vectors.k3k1_out);
    gsl_blas_dscal(b32, integration_vectors.k3k2_out);
    vect_add(integration_vectors.k3_in, 4, 2,
	      integration_vectors.k3k1_out, integration_vectors.k3k2_out);
    gsl_blas_dscal(h, integration_vectors.k3_in);
    gsl_vector_add(integration_vectors.k3_in, in_vec);

    (*f) (x0 + a3 * h, integration_vectors.k3_in,
	  integration_vectors.k3_out);
	  
    //////////////
    // k4
    /////////////
    gsl_vector_memcpy(integration_vectors.k4k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k4k3_out,
		      integration_vectors.k3_out);
    gsl_blas_dscal(b41, integration_vectors.k4k1_out);
    gsl_blas_dscal(b43, integration_vectors.k4k3_out);
    vect_add(integration_vectors.k4_in, 4, 2,
	      integration_vectors.k4k1_out, integration_vectors.k4k3_out);
    gsl_blas_dscal(h, integration_vectors.k4_in);
    gsl_vector_add(integration_vectors.k4_in, in_vec);

    (*f) (x0 + a4 * h, integration_vectors.k4_in,
	  integration_vectors.k4_out);
	  
    //////////////
    // k5
    /////////////
    gsl_vector_memcpy(integration_vectors.k5k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k5k3_out,
		      integration_vectors.k3_out);
    gsl_vector_memcpy(integration_vectors.k5k4_out,
		      integration_vectors.k4_out);
    gsl_blas_dscal(b51, integration_vectors.k5k1_out);
    gsl_blas_dscal(b53, integration_vectors.k5k3_out);
    gsl_blas_dscal(b54, integration_vectors.k5k4_out);
    vect_add(integration_vectors.k5_in, 4, 3,
	      integration_vectors.k5k1_out, integration_vectors.k5k3_out,
	      integration_vectors.k5k4_out);
    gsl_blas_dscal(h, integration_vectors.k5_in);
    gsl_vector_add(integration_vectors.k5_in, in_vec);

    (*f) (x0 + a5 * h, integration_vectors.k5_in,
	  integration_vectors.k5_out);

    /////////////
    // k6
    /////////////
    gsl_vector_memcpy(integration_vectors.k6k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k6k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k6k5_out,
		      integration_vectors.k5_out);
    gsl_blas_dscal(b61, integration_vectors.k6k1_out);
    gsl_blas_dscal(b64, integration_vectors.k6k4_out);
    gsl_blas_dscal(b65, integration_vectors.k6k5_out);
    vect_add(integration_vectors.k6_in, 4, 3,
	      integration_vectors.k6k1_out, integration_vectors.k6k4_out,
	      integration_vectors.k6k5_out);
    gsl_blas_dscal(h, integration_vectors.k6_in);
    gsl_vector_add(integration_vectors.k6_in, in_vec);

    (*f) (x0 + a6 * h, integration_vectors.k6_in,
	  integration_vectors.k6_out);

    //////////////
    // k7
    /////////////
    gsl_vector_memcpy(integration_vectors.k7k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k7k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k7k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k7k6_out,
		      integration_vectors.k6_out);
    gsl_blas_dscal(b71, integration_vectors.k7k1_out);
    gsl_blas_dscal(b74, integration_vectors.k7k4_out);
    gsl_blas_dscal(b75, integration_vectors.k7k5_out);
    gsl_blas_dscal(b76, integration_vectors.k7k6_out);
    vect_add(integration_vectors.k7_in, 4, 4,
	      integration_vectors.k7k1_out, integration_vectors.k7k4_out,
	      integration_vectors.k7k5_out, integration_vectors.k7k6_out);
    gsl_blas_dscal(h, integration_vectors.k7_in);
    gsl_vector_add(integration_vectors.k7_in, in_vec);

    (*f) (x0 + a7 * h, integration_vectors.k7_in,
	  integration_vectors.k7_out);

    //////////////
    // k8
    /////////////
    gsl_vector_memcpy(integration_vectors.k8k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k8k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k8k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k8k7_out,
		      integration_vectors.k7_out);
    gsl_blas_dscal(b81, integration_vectors.k8k1_out);
    gsl_blas_dscal(b85, integration_vectors.k8k5_out);
    gsl_blas_dscal(b86, integration_vectors.k8k6_out);
    gsl_blas_dscal(b87, integration_vectors.k8k7_out);
    vect_add(integration_vectors.k8_in, 4, 4,
	      integration_vectors.k8k1_out, integration_vectors.k8k5_out,
	      integration_vectors.k8k6_out, integration_vectors.k8k7_out);
    gsl_blas_dscal(h, integration_vectors.k8_in);
    gsl_vector_add(integration_vectors.k8_in, in_vec);

    (*f) (x0 + a8 * h, integration_vectors.k8_in,
	  integration_vectors.k8_out);

    //////////////
    // k9
    /////////////
    gsl_vector_memcpy(integration_vectors.k9k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k9k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k9k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k9k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k9k7_out,
		      integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k9k8_out,
		      integration_vectors.k8_out);
    gsl_blas_dscal(b91, integration_vectors.k9k1_out);
    gsl_blas_dscal(b94, integration_vectors.k9k4_out);
    gsl_blas_dscal(b95, integration_vectors.k9k5_out);
    gsl_blas_dscal(b96, integration_vectors.k9k6_out);
    gsl_blas_dscal(b97, integration_vectors.k9k7_out);
    gsl_blas_dscal(b98, integration_vectors.k9k8_out);
    vect_add(integration_vectors.k9_in, 4, 6,
	      integration_vectors.k9k1_out, integration_vectors.k9k4_out,
	      integration_vectors.k9k5_out, integration_vectors.k9k6_out,
	      integration_vectors.k9k7_out, integration_vectors.k9k8_out);

    gsl_blas_dscal(h, integration_vectors.k9_in);
    gsl_vector_add(integration_vectors.k9_in, in_vec);
    (*f) (x0 + a9 * h, integration_vectors.k9_in,
	  integration_vectors.k9_out);

    //////////////
    // k10
    /////////////
    gsl_vector_memcpy(integration_vectors.k10k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k10k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k10k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k10k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k10k7_out,
		      integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k10k8_out,
		      integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k10k9_out,
		      integration_vectors.k9_out);
    gsl_blas_dscal(b10_1, integration_vectors.k10k1_out);
    gsl_blas_dscal(b10_4, integration_vectors.k10k4_out);
    gsl_blas_dscal(b10_5, integration_vectors.k10k5_out);
    gsl_blas_dscal(b10_6, integration_vectors.k10k6_out);
    gsl_blas_dscal(b10_7, integration_vectors.k10k7_out);
    gsl_blas_dscal(b10_8, integration_vectors.k10k8_out);
    gsl_blas_dscal(b10_9, integration_vectors.k10k9_out);
    vect_add(integration_vectors.k10_in, 4, 7,
	      integration_vectors.k10k1_out, integration_vectors.k10k4_out,
	      integration_vectors.k10k5_out, integration_vectors.k10k6_out,
	      integration_vectors.k10k7_out, integration_vectors.k10k8_out,
	      integration_vectors.k10k9_out);
    gsl_blas_dscal(h, integration_vectors.k10_in);
    gsl_vector_add(integration_vectors.k10_in, in_vec);
    (*f) (x0 + a10 * h, integration_vectors.k10_in,
	  integration_vectors.k10_out);

    //////////////
    // k11
    /////////////
    gsl_vector_memcpy(integration_vectors.k11k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k11k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k11k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k11k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k11k7_out,
		      integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k11k8_out,
		      integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k11k9_out,
		      integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k11k10_out,
		      integration_vectors.k10_out);
    gsl_blas_dscal(b11_1, integration_vectors.k11k1_out);
    gsl_blas_dscal(b11_4, integration_vectors.k11k4_out);
    gsl_blas_dscal(b11_5, integration_vectors.k11k5_out);
    gsl_blas_dscal(b11_6, integration_vectors.k11k6_out);
    gsl_blas_dscal(b11_7, integration_vectors.k11k7_out);
    gsl_blas_dscal(b11_8, integration_vectors.k11k8_out);
    gsl_blas_dscal(b11_9, integration_vectors.k11k9_out);
    gsl_blas_dscal(b11_10, integration_vectors.k11k10_out);
    vect_add(integration_vectors.k11_in, 4, 8,
	      integration_vectors.k11k1_out, integration_vectors.k11k4_out,
	      integration_vectors.k11k5_out, integration_vectors.k11k6_out,
	      integration_vectors.k11k7_out, integration_vectors.k11k8_out,
	      integration_vectors.k11k9_out,
	      integration_vectors.k11k10_out);
    gsl_blas_dscal(h, integration_vectors.k11_in);
    gsl_vector_add(integration_vectors.k11_in, in_vec);
    (*f) (x0 + h, integration_vectors.k11_in, integration_vectors.k11_out);

    //////////////
    // k12
    /////////////
    gsl_vector_memcpy(integration_vectors.k12k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k12k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k12k7_out,
		      integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k12k8_out,
		      integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k12k9_out,
		      integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k12k10_out,
		      integration_vectors.k10_out);
    gsl_blas_dscal(b12_1, integration_vectors.k12k1_out);
    gsl_blas_dscal(b12_6, integration_vectors.k12k6_out);
    gsl_blas_dscal(b12_7, integration_vectors.k12k7_out);
    gsl_blas_dscal(b12_8, integration_vectors.k12k8_out);
    gsl_blas_dscal(b12_9, integration_vectors.k12k9_out);
    gsl_blas_dscal(b12_10, integration_vectors.k12k10_out);
    vect_add(integration_vectors.k12_in, 4, 6,
	      integration_vectors.k12k1_out, integration_vectors.k12k6_out,
	      integration_vectors.k12k7_out, integration_vectors.k12k8_out,
	      integration_vectors.k12k9_out,
	      integration_vectors.k12k10_out);
    gsl_blas_dscal(h, integration_vectors.k12_in);
    gsl_vector_add(integration_vectors.k12_in, in_vec);

    (*f) (x0, integration_vectors.k12_in, integration_vectors.k12_out);

    //////////////
    // k13
    /////////////
    gsl_vector_memcpy(integration_vectors.k13k1_out,
		      integration_vectors.k1_out);
    gsl_vector_memcpy(integration_vectors.k13k4_out,
		      integration_vectors.k4_out);
    gsl_vector_memcpy(integration_vectors.k13k5_out,
		      integration_vectors.k5_out);
    gsl_vector_memcpy(integration_vectors.k13k6_out,
		      integration_vectors.k6_out);
    gsl_vector_memcpy(integration_vectors.k13k7_out,
		      integration_vectors.k7_out);
    gsl_vector_memcpy(integration_vectors.k13k8_out,
		      integration_vectors.k8_out);
    gsl_vector_memcpy(integration_vectors.k13k9_out,
		      integration_vectors.k9_out);
    gsl_vector_memcpy(integration_vectors.k13k10_out,
		      integration_vectors.k10_out);
    gsl_vector_memcpy(integration_vectors.k13k12_out,
		      integration_vectors.k12_out);
    gsl_blas_dscal(b13_1, integration_vectors.k13k1_out);
    gsl_blas_dscal(b13_4, integration_vectors.k13k4_out);
    gsl_blas_dscal(b13_5, integration_vectors.k13k5_out);
    gsl_blas_dscal(b13_6, integration_vectors.k13k6_out);
    gsl_blas_dscal(b13_7, integration_vectors.k13k7_out);
    gsl_blas_dscal(b13_8, integration_vectors.k13k8_out);
    gsl_blas_dscal(b13_9, integration_vectors.k13k9_out);
    gsl_blas_dscal(b13_10, integration_vectors.k13k10_out);
    vect_add(integration_vectors.k13_in, 4, 9,
	      integration_vectors.k13k1_out, integration_vectors.k13k4_out,
	      integration_vectors.k13k5_out, integration_vectors.k13k6_out,
	      integration_vectors.k13k7_out, integration_vectors.k13k8_out,
	      integration_vectors.k13k9_out,
	      integration_vectors.k13k10_out,
	      integration_vectors.k13k12_out);
    gsl_blas_dscal(h, integration_vectors.k13_in);
    gsl_vector_add(integration_vectors.k13_in, in_vec);

    (*f) (x0, integration_vectors.k13_in, integration_vectors.k13_out);

    //////////////
    // out_vec
    /////////////
    vect_add(integration_vectors.c_1_11_vec, 4, 2,
	      integration_vectors.k1_out, integration_vectors.k11_out);
    gsl_vector_memcpy(integration_vectors.c_6_vec,
		      integration_vectors.k6_out);
    vect_add(integration_vectors.c_7_8_vec, 4, 2,
	      integration_vectors.k7_out, integration_vectors.k8_out);
    vect_add(integration_vectors.c_9_10_vec, 4, 2,
	      integration_vectors.k9_out, integration_vectors.k10_out);
    gsl_blas_dscal(c_1_11, integration_vectors.c_1_11_vec);
    gsl_blas_dscal(c6, integration_vectors.c_6_vec);
    gsl_blas_dscal(c_7_8, integration_vectors.c_7_8_vec);
    gsl_blas_dscal(c_9_10, integration_vectors.c_9_10_vec);
    vect_add(integration_vectors.c_tot_vec, 4, 4,
	      integration_vectors.c_1_11_vec, integration_vectors.c_6_vec,
	      integration_vectors.c_7_8_vec,
	      integration_vectors.c_9_10_vec);

    gsl_blas_dscal(h, integration_vectors.c_tot_vec);
    gsl_vector_add(integration_vectors.c_tot_vec, in_vec);
    gsl_vector_memcpy(out_vec, integration_vectors.c_tot_vec);

    //////////////
    // err_factor
    /////////////
    gsl_vector_memcpy(integration_vectors.ek12,
		      integration_vectors.k12_out);
    gsl_vector_memcpy(integration_vectors.ek13,
		      integration_vectors.k13_out);
    gsl_blas_dscal(-1, integration_vectors.ek12);
    gsl_blas_dscal(-1, integration_vectors.ek13);

    vect_add(integration_vectors.err_vec, 4, 4,
	      integration_vectors.k1_out, integration_vectors.k11_out,
	      integration_vectors.ek12, integration_vectors.ek13);




    return err_factor * gsl_blas_dnrm2(integration_vectors.err_vec);
}


////////////////
// double_pendulum_eom is the equation of motion for a double pendulum.
//  Ultimately, this should go in an external file which holds the equations of motion.
//  Arguments:
//          t: the current time, double
//          in_state: the state vector at t, gsl_vector
//          out_state: The derivative of in_state at t, gsl_vector
// No return
///////////
void double_pendulum_eom(double t, gsl_vector * in_state,
			 gsl_vector * out_state)
{

    double Th1, Th1_d, Th2, Th2_d;
    Th1 = gsl_vector_get(in_state, 0);
    Th1_d = gsl_vector_get(in_state, 1);
    Th2 = gsl_vector_get(in_state, 2);
    Th2_d = gsl_vector_get(in_state, 3);

    double a1, a2, a3, a4, Th1_dd;
    double b1, b2, b3, b4, Th2_dd;

    a1 = GRAVITY * (sin(Th2) * cos(Th1 - Th2) - 2 * sin(Th1));
    a2 = -(Th2_d * Th2_d + Th1_d * Th1_d * cos(Th1 - Th2));
    a3 = sin(Th1 - Th2);
    a4 = (2 - cos(Th1 - Th2) * cos(Th1 - Th2));
    Th1_dd = (a1 + (a2 * a3)) / a4;

    b1 = 2 * GRAVITY * (sin(Th1) * cos(Th1 - Th2) - sin(Th2));
    b2 = 2 * Th1_d * Th1_d + Th2_d * Th2_d * cos(Th1 - Th2);
    b3 = sin(Th1 - Th2);
    b4 = (2 - cos(Th1 - Th2) * cos(Th1 - Th2));

    Th2_dd = (b1 + (b2 * b3)) / b4;

    gsl_vector_set(out_state, 0, Th1_d);
    gsl_vector_set(out_state, 1, Th1_dd);
    gsl_vector_set(out_state, 2, Th2_d);
    gsl_vector_set(out_state, 3, Th2_dd);
}


/////////////////////////
//   Helper functions  //
/////////////////////////

/////////////////////////////////////////////////////
// vect_add adds arbitrary numbers of gsl_vectors 
// this is used to calculate the k1...k13
// Arguments:
//        sum: A pointer to the gsl_vector which will hold the sum of the vectors, gsl_vector
//        vect_len: the length of the vectors, int
//        count: the number of vectors, int
//        ...: the vectors to be added, gsl_vector
// Returns:
//        0 if successful
//        No failure case yet
/////////////////////////////////////////////////////
static int *vect_add(gsl_vector * sum, int vect_len, int count, ...)
{
    va_list p;
    int i;
    gsl_vector *tmp_vec;
    va_start(p, count);

    for (i = 0; i < count; i++) {
	tmp_vec = va_arg(p, gsl_vector *);
	gsl_blas_daxpy(1, tmp_vec, sum);
    }
    va_end(p);
    return 0;
}



///////////////////
// integration_vector_init initializes the integration_vectors struct
// integration_vector_free frees the struct
// No arguments
////////////////////
static void integration_vector_init()
{
    init_count += 1;
    int i, Nelt;

    gsl_vector **p;		// pointer to a vector

    Nelt = sizeof(integration_vectors) / sizeof(integration_vectors.k1_in);	// number of elements in x.

    p = &integration_vectors.k1_in;	// pointing to the first vector in the struct

    for (i = 0; i < Nelt; i++, p++) {	// p[0] points to integration_vectors.k1_in,
	*p = gsl_vector_calloc(4);	// p[1] points to integration_vectors.k2_in, etc...
    }
}


static void integration_vector_free()
{
    int i, Nelt;
    gsl_vector **p;

    Nelt = sizeof(integration_vectors) / sizeof(integration_vectors.k1_in);	// number of elements in x.

    p = &integration_vectors.k1_in;
    for (i = 0; i < Nelt; i++, p++) {
	gsl_vector_free(*p);
    }
}

//////////////////////
// print_vec prints a gsl_vector
// used for troubleshooting
//
// Argument:
//      vec: a gsl_vector pointer
//
//////////////////////
static void print_vec(gsl_vector * vec)
{
    for (int i = 0; i < 4; i++) {
    }
}

/////////////////////
// populate_history populates the history array
// which is the output of the integration.
// The history array has a shape nsteps * (1 + PhaseSpaceSize)
// for some arbitrary size of a phase space.
// the 0th column is used for the timestamp of that step.
//
// Arguments:
//      hist_len: The number of time steps, integer
//      history: The history array to be populated, double
//      step: The step being populated, int
//      clock: The value of the clock at the current step
//      state: The state at the clock time, gsl_vector
//
// Returns:
//      0 if successful
//      No failure case yet.
///////////////////
static int populate_history(int hist_len,
		      double history[hist_len][5],
		      int step, double clock, gsl_vector * state)
{
    history[step][0] = clock;
    for (int i = 1; i < 5; i++) {
	history[step][i] = gsl_vector_get(state, i - 1);
    }
    return 0;
}

/////////////////
// arr2vec converts a regular ol' array to a gsl_vector
//
// Arguments:
//      len: The length of the array and vector, integer
//      in_arr:  The array being converted, double
//      out_vec:  A pointer to the gsl_vector being populated, gsl_vector
//
// Returns:
//      0 if successful
//      No failure case yet
////////////////
static int arr2vec(int len, double in_arr[len], gsl_vector * out_vec)
{
    for (int i = 0; i < len; i++) {
	gsl_vector_set(out_vec, i, in_arr[i]);
    }
    return 0;
}

int main()
{
    return 0;
}
