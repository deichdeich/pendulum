#include <stdio.h>

double * arr_fucker(double in_arr[], int len)
{   
    for(int i = 0; i < len; i++){
        double v = in_arr[i];
        in_arr[i] = i*v+4;
    }
    return in_arr;
}