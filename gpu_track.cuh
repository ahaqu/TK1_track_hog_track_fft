#ifndef GPU_TRACK_H
#define GPU_TRACK_H

#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

#define eps 0.0001
#define PI 3.1415926535897931

//extern "C" void print_hello_world();

extern "C" void print_hello_world();


extern "C" float* cu_hog( float *I, int h, int w, int nCh, int sBin, int oBin, int oGran );
/*
 extern "C" void cu_compGradImg1( double *I, double *G, int *O, int h, int w, int nCh, int oBin );
 extern "C" void cu_compHist_shell(int h, int w, int sBin, int oBin, int oGran, double *G, int *O, double *hist);
extern "C" void cu_compnorm_shell(int oBin, int nb, double* hist, double* norm);
extern "C" void cu_compmxH_shell(double* mxH, double* norm, int* out,double* hist, int hb, int nb,int oBin, const int outp);
*/

//extern "C" void cu_compFFT_forward(float* raw_data, int width, int height, int flag, cufftComplex *odata, cufftHandle plan);

#endif
