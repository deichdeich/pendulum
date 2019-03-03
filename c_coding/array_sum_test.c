#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <stdarg.h>

/*compiling commands:
1. Compile without linking (wtf is linking):
gcc -Wall -I/usr/local/include -c array_sum_test.c -o array_sum_test.o

2: Link to gsl:
gcc -L/usr/local/lib array_sum_test.o -lgsl -o array_sum_test.e

To run, do ./array_sum_test.e

*/

gsl_vector * _vect_add(int vect_len, int count, ...);

int main(void){
    gsl_vector *v1 = gsl_vector_alloc(5);
    gsl_vector *v2 = gsl_vector_alloc(5);
    gsl_vector *v3 = gsl_vector_alloc(5);
    for (int i = 0; i<5; i++){
        gsl_vector_set(v1, i, i*M_PI + 1);
        gsl_vector_set(v2, i, 2*M_PI*i);
    }
    gsl_vector_memcpy(v3, v2);
    gsl_vector_add(v3, v1);
    for (int i = 0; i<5; i++){
        printf("%f + %f = %f\n", gsl_vector_get(v1, i),
                               gsl_vector_get(v2, i),
                               gsl_vector_get(v3, i));
    }
    gsl_vector * blah;
    blah = variad(5, 4, v3, v2, v1, v3);
    for (int i = 0; i < 5; i++){
    printf("%f\n", gsl_vector_get(blah, i));
    }
    return 0;
}

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
gsl_vector * _vect_add(int vect_len, int count, ...){
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

/*
int vect_fucker(int count, gsl_vector * v, ...){
    va_list vects;
    int i;
    for (i = 0; i<count, i++){
    gsl_vector_add()
    }
    return 0;
}
*/