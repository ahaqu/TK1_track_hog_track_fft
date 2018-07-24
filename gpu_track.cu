#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "time.h"
#include "gpu_track.cuh"
#include "cufft.h"
#include "device_functions.h"

using namespace std;

#define THREADS_NUM 64
#define BLOCKS_NUM 1

__global__ void cu_compHist(int h, int w, int sBin, int oBin, int oGran, float *G, int *O, float *hist);
__global__ void cu_compnorm(int oBin, int nb, float* hist, float* norm);
__global__ void cu_compmxH(float* mxH, float* norm, int* out,float* hist, int hb, int nb,int oBin, const int outp);
 extern "C" void cu_compGradImg1( float *I,float* I_tmp, float *G, int *O, int h, int w, int nCh, int oBin );

 extern "C" __global__ void SaXPY(float a, float* X_d, float* Y_d, int n)
{
     if (threadIdx.x < n)
        Y_d[threadIdx.x] = a * X_d[threadIdx.x] + Y_d[threadIdx.x];
 }

extern "C" void print_hello_world()
 {
	 cout << "hello world version 2" << endl;
 }


//__global__ void cu_compHist(int h, int w, int sBin, int oBin, int oGran, double *G, int *O, double *hist)
__global__ void cu_compHist(int h, int w, int sBin, int oBin, int oGran, float *G, int *O, float *hist)
{
	
	//hb是竖着有几个cell，wb是横着有几个cell，h0是在纵向cell内的单位数量，w0是在横向cell内的单位数量，nb是cell的总个数
//	const int hb=h/sBin, wb=w/sBin, h0=hb*sBin-4, w0=wb*sBin-4, nb=wb*hb;
	const int hb=h/sBin, wb=w/sBin, w0=wb*sBin-4, nb=wb*hb;

	__shared__ float hist_cache[34*34*9];

	const int tid = blockIdx.x*792+threadIdx.x;

	if(tid<612)
	{
			for (int qq = 0; qq<17; qq++)
			{
				hist_cache [tid*17+qq] = 0;

			}
	}

//		hist_cache[0] = 0.0;
			__syncthreads();

//	if(hist_cache[0]==0){


		for (int p=0; p<11; p++){

			int xy = tid+p*792*2;
//			int xy = tid*11 + p;
			int x = xy/w0+2;
			int y = xy%w0+2;
			float v=float(*(G+x*h+y)); float o = float(*(O+x*h+y))/float(oGran);
			int o0=int(o); int o1=(o0+1)%oBin; float od0=o-o0, od1=1.0-od0;

			float xb = (float(x)+.5)/float(sBin)-0.5; 
			int xb0=int(xb);
			float yb = (float(y)+.5)/float(sBin)-0.5;
			int yb0=int(yb);
			float xd0=xb-xb0, xd1=1.0-xd0; float yd0=yb-yb0, yd1=1.0-yd0;
			float *dst = hist_cache + xb0*hb + yb0;

			*(dst+o0*nb)      += od1*xd1*yd1*v;
			*(dst+hb+o0*nb)   += od1*xd0*yd1*v;
			*(dst+1+o0*nb)    += od1*xd1*yd0*v;
			*(dst+hb+1+o0*nb) += od1*xd0*yd0*v;

			*(dst+o1*nb)      += od0*xd1*yd1*v;
			*(dst+hb+o1*nb)   += od0*xd0*yd1*v; 
			*(dst+1+o1*nb)    += od0*xd1*yd0*v;
			*(dst+hb+1+o1*nb) += od0*xd0*yd0*v;

		}
//	}

		__syncthreads();

//		if (blockIdx.x == 0){
			if(tid<612)
			{
				for (int qq = 0; qq<17; qq++)
				{
					*(hist + tid*17+qq) = hist_cache[tid*17+qq];
//					*(hist + tid*17+qq) = *(hist_cache+tid*17+qq);

				}
			}
//		}
/*
		__syncthreads();

		if (blockIdx.x == 1){
			if((tid-792)<612)
			{
				for (int qq = 0; qq<17; qq++)
				{
					*(hist + (tid-792)*17+qq) += hist_cache[(tid-792)*17+qq];

				}
			}
		}
*/
}


// oBin = 9
// nb = 34*34
__global__ void cu_compnorm(int oBin, int nb, float* hist, float* norm)
{

	const int tid = threadIdx.x;

//	const int block_size = (oBin*nb) / (THREADS_NUM-1);
	const int block_size = (oBin*nb) / 612;

//	double *dst=norm, *end=norm+nb;

//	double *src = hist;

	//	for( int o=0; o<oBin; o++ ) {
//	for( int oi= tid *block_size; oi<((tid==(THREADS_NUM-1)) ? oBin*nb:(tid+1)*block_size); oi++ ) {
	for(int oi = tid*block_size; oi<(tid+1)*block_size;oi++){
//		int o = oi/oBin;
//		int nb = oi%oBin;
		float* dst = norm + oi;
		float* src = hist + oi;
		*(dst)+=(*src)*(*src);
	}
}

__global__ void cu_compmxH(float* mxH, float* norm, int* out,float* hist, int hb, int nb,int oBin, const int outp)
{
	//	double *H = mxH;

	const int tid = threadIdx.x;

//	const int block_size = (out[1]*out[0]) / (THREADS_NUM-1);

	//	for( int x=0; x<out[1]; x++ ) 
	//		for( int y=0; y<out[0]; y++ ) {
	//	for( int xy=tid*block_size; xy<((tid==(THREADS_NUM-1)) ? out[1]*out[0]:(tid+1)*block_size); xy++ ) {
	//		int x = xy/out[1];
	//		int y = xy%out[1];
	int x = tid/32;
	int y = tid%32;
	float *dst=mxH+x*out[0]+y; 
	float *src, *p, n;

	p = norm + (x)*hb + (y);
	n = 1.0/sqrt(*p + *(p+1) + *(p+hb) + *(p+hb+1) + eps);
	src = hist + (x+1)*hb + (y+1);
	for( int o=0; o<oBin; o++ ) {
		*dst=min(float(*src*n), 0.2); dst+=outp; src+=nb;
	}
	//	}
}


__global__ void cu_compGradImg_part(int oBin, int nCh, float* I, float* G, int* O, int h, int w)
{

	const int tid = blockIdx.x * 782 + threadIdx.x;
	int h_tmp = h+2;
	int w_tmp = w+2;
	int x_tmp, y_tmp;

	__shared__ float I_tmp[138*70];

	int mem_block_count = 2;


for ( int mem_block =0; mem_block<mem_block_count; mem_block++)
{

	int tmp_block = 136*69/(782);

	int start_x = mem_block*68;


	for (int xy=(tid*tmp_block); xy<((tid+1)*tmp_block); xy++)
	{
		int x = 1 + xy/136;
		int y = 1 + xy%136;
		I_tmp[x*w_tmp+y] = I[(x-1+start_x)*w+y-1];
	}
	__syncthreads();


	if (tid <136)
	{
		x_tmp = mem_block?69:0;
		int y = tid+1;
		if(mem_block)
			I_tmp[x_tmp*h_tmp+y] = 2*I_tmp[(x_tmp-1)*h_tmp+y]-I_tmp[(x_tmp-2)*h_tmp+y];
		else
			I_tmp[x_tmp*w_tmp+y] = 2*I_tmp[(x_tmp+1)*w_tmp+y]-I_tmp[(x_tmp+2)*w_tmp+y];
	}

	if((tid>=136)&&(tid<205)){
		y_tmp =0;
		int x=tid-136+1;
		I_tmp[x*w_tmp+y_tmp] = 2*I_tmp[x*w_tmp+y_tmp+1] - I_tmp[x*w_tmp+y_tmp+2];
	}

	if((tid>=205)&&(tid<274)){
		y_tmp =w+1;
		int x = tid-205+1;
		I_tmp[x*w_tmp+y_tmp] = 2*I_tmp[x*w_tmp+y_tmp-1] - I_tmp[x*w_tmp+y_tmp-2];

	}
	__syncthreads();



	//	const int block_size = ((w-2)*(h-2))/(1024-1);
	const int block_size = ((w)*(68))/(544);

if(tid<544){
	// compute gradients for each channel, pick strongest gradient
	int y, x;
	float *I1, v, dx, dy, dx1, dy1, v1;

	// centered differences on interior points
	for (int xy = tid*block_size; xy<(tid+1)*block_size; xy++)
	{
		x = xy/(w);
		y = xy%(w);

		I1 = I_tmp + (x+1)*(w+2) + y+1; 
		dy1 = (*(I1+1)-*(I1-1)); 
		dx1 = (*(I1+w+2)-*(I1-w-2)); 
		v1=dx1*dx1+dy1*dy1;

		v=v1; 
		dx=dx1; 
		dy=dy1;

		*(G+ (start_x+x)*w+y)=sqrt(v); 
		float o = fabs(atan2(dy,dx));
		int index = (int)((float)(o)/(PI/90)+0.5);
		index %= 90;
		*(O+(start_x+x)*w+y)=index;

	}
	}
__syncthreads();
}
}

// compute HOG features

extern "C" float* cu_hog( float *I, int h, int w, int nCh, int sBin, int oBin, int oGran ) {
	// compute gradient magnitude (*2) and orientation for each location in I

	long begin=clock();
	
//	const int hb=h/sBin, wb=w/sBin, h0=hb*sBin, w0=wb*sBin, nb=wb*hb;
	const int hb=h/sBin, wb=w/sBin, nb=wb*hb;
	float *G;
	cudaMalloc( &G, h * w * sizeof( float ) ); 
	cudaMemset(G,0,sizeof(float)*h*w);

	int *O;
	cudaMalloc( &O, h*w*sizeof( int ));
	cudaMemset(O,0,sizeof(int)*h*w);
	//I是图片，nch固定
	cu_compGradImg_part<<<1,782>>>(oBin, nCh, I, G, O, h, w);

	begin = clock();
	float *hist;
	cudaMalloc(&hist,nb*oBin*sizeof(float));
	cudaMemset(hist,0,sizeof(float)*nb*oBin);

	cu_compHist<<<1,792>>>(h,w, sBin,oBin,oGran, G, O, hist);

	cudaFree(G);
	cudaFree(O);


	float *norm;
	cudaMalloc(&norm,nb*sizeof(float));
	cudaMemset(norm,0,sizeof(float)*nb);

	cu_compnorm<<<BLOCKS_NUM,612>>>(oBin, nb, hist, norm);

	// compute normalized values (4 different normalizations per block)
	const int out[3] = { max(hb-2, 0), max(wb-2, 0), oBin*4 }; const int outp=out[0]*out[1];

	int * gpu_out;
	cudaMalloc(&gpu_out,3*sizeof(int));
	cudaMemset(gpu_out,0,sizeof(int)*3);
	cudaMemcpy(gpu_out, out, sizeof(int)*3,cudaMemcpyHostToDevice);

	float *mxH;
	cudaMalloc(&mxH,out[0]*out[1]*oBin*4*sizeof(float));
	cudaMemset(mxH,0,sizeof(float)*out[0]*out[1]*oBin*4);

	cu_compmxH<<<BLOCKS_NUM,1024>>>(mxH,norm, gpu_out, hist, hb, nb,oBin, outp);


	cudaFree(hist);
	cudaFree(norm);
	cudaFree(gpu_out);
	return mxH;

}



__global__ void cu_memcpy(float* cu_raw, float* odata, int width, int height)
{

	const int tid = blockIdx.x * 1024 + threadIdx.x;

 	for (int km=tid*16; km<(tid+1)*16; km++)
	{
		odata[km*2] = cu_raw[km];
		odata[km*2+1] = 0;
	}

}


 extern "C" void cu_compFFT_forward(float* raw_data, float* dst_data, int width, int height, int flag, cufftComplex *odata, cufftHandle plan, float* cu_raw)
{
//	clock_t start = clock();
//	cufftComplex * odata;

//	long t1 = clock();

//	float* return_data = new float[height*width*2];
/*
	if(  odata == NULL)
	{
		cout<<"odata is NULL"<<endl;
		cout<<" mem size is "<< sizeof(cufftComplex)*height*width<<endl;
		exit(0);
	}
		cout<<" mem size is "<< sizeof(cufftComplex)*height*width<<endl;
//	cudaMalloc((void**)&odata,sizeof(cufftComplex)*height*width);


//	cufftHandle plan;
	int rc = cufftPlan2d(&plan,height,width,CUFFT_C2C);
	if (rc != CUFFT_SUCCESS)
	{
		cout<< "rc from plan is "<<rc<<endl;
	
		exit(0);
	}
*/
//	cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_PADDING);
//	cout<<"------------------------fft_prepare"<<clock()-t1<<endl;
//	t1 = clock();


//	cudaMemcpy (odata,raw_data,sizeof(cufftComplex)*height*width,cudaMemcpyHostToDevice);
//	cufftExecC2C(plan,odata,odata,CUFFT_INVERSE );
	if (flag==0)
	{
//		float* cu_raw;
//		cudaMalloc((void**)&cu_raw,sizeof(float)*height*width);

		cudaMemcpy (cu_raw,raw_data,sizeof(float)*height*width,cudaMemcpyHostToDevice);
		
		cu_memcpy<<<1,1024>>>((float*)cu_raw,(float*)odata,width,height);

		int rc = cufftExecC2C(plan,(cufftComplex*)odata,(cufftComplex*)odata,CUFFT_FORWARD );
		if(rc!=CUFFT_SUCCESS)
		{
			cout<<"forward*****************************************************"<<endl;
			cout<<"RC is "<<rc<<endl;
//			exit(0);
		}
//		cudaFree(cu_raw);
//		else
//			cout<<"forward.........................."<<endl;
	}
	else if(flag == 1)
	{
		cudaMemcpy (odata,raw_data,sizeof(cufftComplex)*height*width,cudaMemcpyHostToDevice);

		int rc = cufftExecC2C(plan,(cufftComplex*)odata,(cufftComplex*)odata,CUFFT_INVERSE );
		if(rc!=CUFFT_SUCCESS)
		{
			cout<<"inverse*****************************************************"<<endl;
			cout<<"RC is "<<rc<<endl;
//			exit(0);
		}
//		else
//			cout<<"inverse.........................."<<endl;
	}
	cudaMemcpy (dst_data,odata,sizeof(cufftComplex)*height*width,cudaMemcpyDeviceToHost);

//	cout<<"_-----------------------fft exec"<<clock()-t1<<endl;
//	t1 = clock();


//	cudaDeviceSynchronize();

//	cufftDestroy(plan);

//	cudaFree(odata);

//	cout<<"------------------------fft end"<<clock()-t1<<endl;
	
//	cout<< " FFT_forward..............................."<< clock()-start <<endl;

}


__global__ void cu_comp_gaussianKernel( float xx , float yy , float *xy , double sigma  , float *output, int nWidth, int nHeigth ) {
	
	const int tid = blockIdx.x * 1024 + threadIdx.x;

	float sigma2 = sigma*sigma;

//	__shared__ float output_cache = [1024];

	float N = nWidth*nHeigth;
	int block_size = N/1024;
	for( int index = tid*block_size; index < (tid+1)*block_size; index++ ) {
			
		float value = (xx + yy - 2*xy[index])/N;
		float v = exp(-max(0.0, value) / sigma2);
//		output_cache[index] = v;
		output[index] = v;
	}
	
}

extern "C" void cu_gaussianKernel( float xx , float yy , float *xy , double sigma  , float *output, int nWidth, int nHeight ) 
{

	float* cu_xy;
	float* cu_output;

	cudaMalloc((void**)&cu_xy,sizeof(float)*nHeight*nWidth);
	cudaMalloc((void**)&cu_output,sizeof(float)*nHeight*nWidth);

	cudaMemcpy (cu_xy,xy,sizeof(float)*nHeight*nWidth,cudaMemcpyHostToDevice);

	cu_comp_gaussianKernel<<<1,1024>>>(xx,yy,cu_xy,sigma,cu_output,nWidth,nHeight);

	cudaMemcpy (output,cu_output,sizeof(float)*nHeight*nWidth,cudaMemcpyDeviceToHost);

	cudaFree(cu_xy);
	cudaFree(cu_output);

}






