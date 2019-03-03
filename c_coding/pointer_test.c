#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int pointer_test_wrong(int a, int b){
    
    a += 1;
    b += 2;
    
    return(0);
}


int pointer_test_right(int *a, int *b){
    
    *a += 1;
    *b += 2;

    return(0);
}

int pointer_test_function(int (*f)(int, int), int x, int y, int *z){

    *z = (*f)(x, y);
    
    return 0;
}

int test_func(int x, int y){

    int ret = x + y;
    return ret;

}

int f(int n, int (*h)(int)){
      
    return (*h)(n);
    
    }
    
int g(int n){

    return n+1;

    }