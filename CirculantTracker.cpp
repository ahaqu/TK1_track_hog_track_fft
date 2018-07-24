#include "CirculantTracker.h"
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <iostream>
//#include <conio.h>
#include <math.h>
#include <fstream>

#define LINUX

#ifdef LINUX
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda.h" 
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda_runtime.h"
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cufft.h"
#else
#include "cuda.h" 
#include "cuda_runtime.h"
#include "cufft.h"
#endif

#define LOG if(0) cout

#define TC //

//#define FFT

using namespace std;
using namespace cv;	


extern "C" float* cu_hog( float *I, int h, int w, int nCh, int sBin, int oBin, int oGran );
extern "C" void cu_compFFT_forward(float* raw_data, float* dst_data,int width, int height, int flag, cufftComplex *odata, cufftHandle plan, float *cu_raw);
extern "C" void cu_gaussianKernel( float xx , float yy , float *xy , double sigma  , float *output, int nWidth, int nHeigth );
/*
extern "C" void cu_compGradImg1( double *I, double *G, int *O, int h, int w, int nCh, int oBin );
 extern "C" void cu_compHist_shell(int h, int w, int sBin, int oBin, int oGran, double *G, int *O, double *hist);
 extern "C" void cu_compnorm_shell(int oBin, int nb, double* hist, double* norm);
 extern "C" void cu_compmxH_shell(double* mxH, double* norm, int* out,double* hist, int hb, int nb,int oBin, const int outp);
 */

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

#ifndef M_PI
#define M_PI 3.1415926
#endif

#define eps 0.0001

#ifndef PI
#define PI 3.1415926535897931
#endif

int count_fft = 0;


//template <class T>
//T**	New2DPointer(int n1D, int n2D)
//{
//	T**		pp;
//	typedef		T*	T_P;
//	int		i, j;
//	if(n1D <= 0 || n2D <= 0)
//		return 0;
//
//	pp = new T_P[n1D];
//	if(!pp)
//		return 0;
//	for(i=0; i<n1D; i++)
//	{
//		pp[i] = new T[n2D];
//		if(!pp[i])
//		{
//			for(j=0; j<i; j++)
//			{
//				delete[] pp[j];
//			}
//			delete[] pp;
//			return 0;
//		}
//	}
//	return pp;
//}
//
//template <class T>
//void	Delete2DPointer(T **pp, int n1D)
//{
//	int		i;
//	if(pp == 0)
//		return;
//	for(i=0; i<n1D; i++)
//	{
//		if(pp[i])
//			delete[] pp[i];
//	}
//	delete[]	pp;
//}

//using namespace std;

int					compOrient1( double dx, double dy, double *ux, double *uy, int oBin ) {
	if(oBin<=1) return 0; int o0=0, o1=oBin-1;
	double s0=fabs(ux[o0]*dx+uy[o0]*dy);
	double s1=fabs(ux[o1]*dx+uy[o1]*dy);
	while( 1 ) {
		if(o0==o1-1) { return ((s0>s1) ? o0 : o1); }
		if( s0<s1 ) {
			o0+=(o1-o0+1)>>1; s0=fabs(ux[o0]*dx+uy[o0]*dy);
		} else {
			o1-=(o1-o0+1)>>1; s1=fabs(ux[o1]*dx+uy[o1]*dy);
		}
	}
}
// compute gradient magnitude (*2) and orientation
void				compGradImg1( double *I, double *G, int *O, int h, int w, int nCh, int oBin ) {
	// compute unit vectors evenly distributed at oBin orientations
	//double *ux = (double*) mxMalloc(oBin*sizeof(double));
	//double *uy = (double*) mxMalloc(oBin*sizeof(double));
	double *ux = new double[oBin];
	double *uy = new double[oBin];
	for( int o=0; o<oBin; o++ ) ux[o]=cos(double(o)/double(oBin)*PI);
	for( int o=0; o<oBin; o++ ) uy[o]=sin(double(o)/double(oBin)*PI);

	// compute gradients for each channel, pick strongest gradient
	int y, x, c; double *I1, v, dx, dy, dx1, dy1, v1;
#define COMPGRAD(x0, x1, rx, y0, y1, ry) { v=-1; for(c=0; c<nCh; c++) { \
	I1 = I + c*h*w + x*h + y; \
	dy1 = (*(I1+y1)-*(I1-y0))*ry; \
	dx1 = (*(I1+x1*h)-*(I1-x0*h))*rx; \
	v1=dx1*dx1+dy1*dy1; if(v1>v) { v=v1; dx=dx1; dy=dy1; }} \
	*(G+x*h+y)=sqrt(v); *(O+x*h+y)=compOrient1(dx, dy, ux, uy, oBin); }

	// centered differences on interior points
	for( x=1; x<w-1; x++ ) for( y=1; y<h-1; y++ ) COMPGRAD(1, 1, 1, 1, 1, 1);

	// uncentered differences along each edge
	x=0;   for( y=1; y<h-1; y++ ) COMPGRAD(0, 1, 2, 1, 1, 1);
	y=0;   for( x=1; x<w-1; x++ ) COMPGRAD(1, 1, 1, 0, 1, 2);
	x=w-1; for( y=1; y<h-1; y++ ) COMPGRAD(1, 0, 2, 1, 1, 1);
	y=h-1; for( x=1; x<w-1; x++ ) COMPGRAD(1, 1, 1, 1, 0, 2);

	// finally uncentered differences at corners
	x=0;   y=0;   COMPGRAD(0, 1, 2, 0, 1, 2);
	x=w-1; y=0;   COMPGRAD(1, 0, 2, 0, 1, 2);
	x=0;   y=h-1; COMPGRAD(0, 1, 2, 1, 0, 2);
	x=w-1; y=h-1; COMPGRAD(1, 0, 2, 1, 0, 2);

	/*mxFree(ux); mxFree(uy);*/
	delete []ux;
	delete []uy;
}

// compute HOG features
double*			hog( double *I, int h, int w, int nCh, int sBin, int oBin, int oGran ) {
	// compute gradient magnitude (*2) and orientation for each location in I
	//double *G = (double*) mxMalloc(h*w*sizeof(double));
	//int *O = (int*) mxMalloc(h*w*sizeof(int));
	double *G = new double[h*w];
	memset(G,0,sizeof(double)*h*w);
	int *O = new int[h*w];
	memset(O,0,sizeof(int)*h*w);
	//I是图片，nch固定
	compGradImg1(I, G, O, h, w, nCh, oBin*oGran);

	// compute gradient histograms use trilinear interpolation on spatial and orientation bins
	const int hb=h/sBin, wb=w/sBin, h0=hb*sBin, w0=wb*sBin, nb=wb*hb;
	//double *hist = (double*) mxCalloc(nb*oBin, sizeof(double));
	double *hist = new double[nb*oBin];
	memset(hist,0,sizeof(double)*nb*oBin);

	//统计直方图到9份
	if( oGran==1 ) 
		for( int x=0; x<w0; x++ ) 
			for( int y=0; y<h0; y++ ) 
			{ // bilinear interp.
				double v=*(G+x*h+y); int o = *(O+x*h+y);
				double xb = (double(x)+.5)/double(sBin)-0.5; int xb0=(xb<0) ? -1 : int(xb);
				double yb = (double(y)+.5)/double(sBin)-0.5; int yb0=(yb<0) ? -1 : int(yb);
				double xd0=xb-xb0, xd1=1.0-xd0; double yd0=yb-yb0, yd1=1.0-yd0;
				double *dst = hist + o*nb + xb0*hb + yb0;
				if( xb0>=0 && yb0>=0     ) *(dst)      += xd1*yd1*v;
				if( xb0+1<wb && yb0>=0   ) *(dst+hb)   += xd0*yd1*v;
				if( xb0>=0 && yb0+1<hb   ) *(dst+1)    += xd1*yd0*v;
				if( xb0+1<wb && yb0+1<hb ) *(dst+hb+1) += xd0*yd0*v;
			} 
	else 
		for( int x=0; x<w0; x++ ) 
			for( int y=0; y<h0; y++ ) 
			{ // trilinear interp.
				double v=*(G+x*h+y); double o = double(*(O+x*h+y))/double(oGran);
				int o0=int(o); int o1=(o0+1)%oBin; double od0=o-o0, od1=1.0-od0;
				double xb = (double(x)+.5)/double(sBin)-0.5; int xb0=(xb<0) ? -1 : int(xb);
				double yb = (double(y)+.5)/double(sBin)-0.5; int yb0=(yb<0) ? -1 : int(yb);
				double xd0=xb-xb0, xd1=1.0-xd0; double yd0=yb-yb0, yd1=1.0-yd0;
				double *dst = hist + xb0*hb + yb0;
				if( xb0>=0 && yb0>=0     ) 
					*(dst+o0*nb)      += od1*xd1*yd1*v;
				if( xb0+1<wb && yb0>=0   ) 
					*(dst+hb+o0*nb)   += od1*xd0*yd1*v;
				if( xb0>=0 && yb0+1<hb   ) 
					*(dst+1+o0*nb)    += od1*xd1*yd0*v;
				if( xb0+1<wb && yb0+1<hb ) 
					*(dst+hb+1+o0*nb) += od1*xd0*yd0*v;
				if( xb0>=0 && yb0>=0     ) 
					*(dst+o1*nb)      += od0*xd1*yd1*v;
				if( xb0+1<wb && yb0>=0   ) 
					*(dst+hb+o1*nb)   += od0*xd0*yd1*v;
				if( xb0>=0 && yb0+1<hb   ) 
					*(dst+1+o1*nb)    += od0*xd1*yd0*v;
				if( xb0+1<wb && yb0+1<hb ) 
					*(dst+hb+1+o1*nb) += od0*xd0*yd0*v;
			}
	delete []G;
	delete []O;
	//mxFree(G); mxFree(O);

	// compute energy in each block by summing over orientations
	//double *norm = (double*) mxCalloc(nb, sizeof(double));
	double *norm = new double[nb];
	memset(norm,0,sizeof(double)*nb);
	for( int o=0; o<oBin; o++ ) {
		double *src=hist+o*nb, *dst=norm, *end=norm+nb;
		while( dst < end ) { *(dst++)+=(*src)*(*src); src++; }
	}

	// compute normalized values (4 different normalizations per block)
	const int out[3] = { max(hb-2, 0), max(wb-2, 0), oBin*4 }; const int outp=out[0]*out[1];
	//mxArray *mxH = mxCreateNumericArray(3, out, mxDOUBLE_CLASS, mxREAL);
	double *mxH = new double[out[0]*out[1]*oBin*4];
	//double mxH[1728];
	memset(mxH,0,sizeof(double)*out[0]*out[1]*oBin*4);
	//double *H = (double*) mxGetPr(mxH);
	double *H = mxH;
	for( int x=0; x<out[1]; x++ ) for( int y=0; y<out[0]; y++ ) {
		double *dst=H+x*out[0]+y; double *src, *p, n;
		for( int x1=1; x1>=0; x1-- ) for( int y1=1; y1>=0; y1-- ) {
			p = norm + (x+x1)*hb + (y+y1);
			n = 1.0/sqrt(*p + *(p+1) + *(p+hb) + *(p+hb+1) + eps);
			src = hist + (x+1)*hb + (y+1);
			for( int o=0; o<oBin; o++ ) {
				*dst=min(*src*n, 0.2); dst+=outp; src+=nb;
			}
		}
	}
	delete []hist;
	delete []norm;
	//mxFree(hist); mxFree(norm);
	return mxH;
}





CirculantTracker::CirculantTracker(void)
{

}


CirculantTracker::~CirculantTracker(void)
{
	if (fft!=NULL)
	{
		delete fft;
	}

	if (templateNew!=NULL)
	{
		delete []templateNew;
	}

	if (templateOld!=NULL)
	{
		delete []templateOld;
	}

	if (cosine!=NULL)
	{
		delete []cosine;
	}

	if (k!=NULL)
	{
		delete []k;
	}

	if (kf!=NULL)
	{
		delete []kf;
	}

	if (alphaf!=NULL)
	{
		delete []alphaf;
	}

	if (newAlphaf!=NULL)
	{
		delete []newAlphaf;
	}

	if (response!=NULL)
	{
		delete []response;
	}

	if (tmpReal0!=NULL)
	{
		delete []tmpReal0;
	}

	if (tmpReal1!=NULL)
	{
		delete []tmpReal1;
	}

	if (tmpFourier0!=NULL)
	{
		delete []tmpFourier0;
	}

	if (tmpFourier1!=NULL)
	{
		delete []tmpFourier1;
	}

	if (tmpFourier2!=NULL)
	{
		delete []tmpFourier2;
	}

	if (gaussianWeight!=NULL)
	{
		delete []gaussianWeight;
	}

	if (gaussianWeightDFT!=NULL)
	{
		delete []gaussianWeightDFT;
	}
	
	if (cufft_odata != NULL)
	{
		cudaFree(cufft_odata);
	}
	
	cufftDestroy(cufft_plan);

}

CirculantTracker::CirculantTracker(float output_sigma_factor_p, float sigma_p, float lambda_p, float interp_factor_p,
	float padding_p ,
	float maxPixelValue_p) {

		output_sigma_factor = output_sigma_factor_p;
		sigma = sigma_p;
		lambda = lambda_p;
		interp_factor = interp_factor_p;
		maxPixelValue = maxPixelValue_p;

		padding = padding_p;

		fft=NULL;
		templateNew=NULL;
		templateOld=NULL;
		cosine=NULL;
		k=NULL;
		kf=NULL;
		alphaf=NULL;
		newAlphaf=NULL;
		response=NULL;
		tmpReal0=NULL;
		tmpReal1=NULL;
		tmpFourier0=NULL;
		tmpFourier1=NULL;
		tmpFourier2=NULL;
		gaussianWeight=NULL;
		gaussianWeightDFT=NULL;

		t_fft = 0;
		t_dense_gause = 0;
		t_subwindow = 0;
		t_hog=0;

		/*workRegionSize = workRegionSize_p;
		workRegionWidth = workRegionSize_p;
		workRegionHeight = workRegionSize_p;*/

		//localPeak.setImage(response);
}

//图像区域 128*128
#ifdef FFT
void CirculantTracker::initialize( unsigned char* image , int imWidth, int imHeight, int x0 , int y0 , int regionWidth , int regionHeight ) {
#else
 void CirculantTracker::initialize( unsigned char* image , int imWidth, int imHeight, int x0 , int y0 , int regionWidth , int regionHeight, int cellSize ) {
#endif
		regionOut.m_width = regionWidth;
		regionOut.m_height = regionHeight;
#ifndef FFT
		m_cellSize = cellSize;
#endif
		m_imWidth = imWidth;
		m_imHeight = imHeight;

		// adjust for padding
		int w = (int)(regionWidth*(1+padding));
		int h = (int)(regionHeight*(1+padding));
		//hog 特征时候的128上下左右加上4，就是136
#ifdef FFT
		w = 128;
		h = 128;
#else
		w = 136;
		h= 136;
#endif




		int cx = x0 + regionWidth/2;
		int cy = y0 + regionHeight/2;

		// save the track location
		regionTrack.m_width = w;
		regionTrack.m_height = h;
		//搜索区域的位置
		regionTrack.m_xMin = cx-w/2;
		regionTrack.m_yMin = cy-h/2;

		//workRegionSize = w;
		//去掉边上的数据
#ifdef FFT
		workRegionWidth = w;
		workRegionHeight = h;
		resizeImages(workRegionWidth,workRegionHeight);
#else
		workRegionWidth = floor(double(w)/cellSize-2);
		workRegionHeight = floor(double(h)/cellSize-2);
		resizeImages(workRegionWidth,workRegionHeight,9);
#endif

		cudaError_t cudaStatus = cudaMalloc((void**)&cufft_odata,sizeof(cufftComplex)*workRegionHeight*workRegionWidth);
		cudaStatus = cudaMalloc((void**)&cufft_rawdata,sizeof(float)*workRegionHeight*workRegionWidth);
		if (cudaStatus != cudaSuccess)
		{
			LOG<< " gout error with "<<cudaStatus<<endl;
			LOG<<"Men ===== "<<sizeof(cufftComplex)*workRegionHeight*workRegionWidth<<endl;
		//	exit(0);
		}
		else
			LOG<< " -99999999999999999999999999999999999999999"<<endl;

		cufftPlan2d(&cufft_plan,workRegionHeight,workRegionWidth,CUFFT_C2C);

		fft = new GeneralFFT_2D(workRegionHeight,workRegionWidth);
		//清零，可以去掉
		
		//cons的加权函数，突出中心区域
		computeCosineWindow(cosine,workRegionWidth,workRegionHeight);
		//高斯加权
		computeGaussianWeights(regionWidth,regionHeight);
#ifdef FFT
		stepX = (w-1)/(float)(workRegionWidth-1);
		stepY = (h-1)/(float)(workRegionHeight-1);
#endif
		//往外输出中心点的位置，可以忽略
		updateRegionOut();

		initialLearning(image, imWidth, imHeight);
		
	}

 //主要计算
 void CirculantTracker::initialLearning( unsigned char* image, int imWidth, int imHeight) {
	 // get subwindow at current estimated target position, to train classifier
	 //提取136*136的图出来
	 get_subwindow(image, templateOld);

	 // Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
	 //计算高斯核函数，得到实域
	 dense_gauss_kernel(sigma, templateOld, templateOld,k);
//	 memcpy(kf,k,sizeof(float)*workRegionHeight*workRegionWidth);
	 //计算平域
//	 fft->realForwardFull(kf);
/* 	for (int km=0; km<workRegionHeight*workRegionWidth; km++)
	{
		kf[km*2] = k[km];
		kf[km*2+1] = 0;
	}
*/
	cu_compFFT_forward(k,kf, workRegionWidth, workRegionHeight,0, cufft_odata, cufft_plan,cufft_rawdata);

	 // new_alphaf = yf ./ (fft2(k) + lambda);   %(Eq. 7)
	 //计算特征值
	 computeAlphas(gaussianWeightDFT, kf, lambda, alphaf, workRegionWidth,workRegionHeight);
 }

 void CirculantTracker::computeCosineWindow( float* cosine,int nWidth, int nHeight ) {
	 float *cosX = new float[ nWidth ];
	 for( int x = 0; x < nWidth; x++ ) {
		 cosX[x] = 0.5*(1 -cos( 2.0*M_PI*x/(nWidth-1) ));
	 }
	 for( int y = 0; y < nHeight; y++ ) {
		 int index =  y*nWidth;
		 double cosY = 0.5*(1 - cos( 2.0*M_PI*y/(nHeight-1) ));
		 for( int x = 0; x < nWidth; x++ ) {
			 cosine[index++] = cosX[x]*cosY;
		 }
	 }
	 delete []cosX;
 }

void CirculantTracker::computeGaussianWeights( int width,int height ) {
	// desired output (gaussian shaped), bandwidth proportional to target size
#ifdef FFT
	float output_sigma = sqrt(float(width*height)) * output_sigma_factor;
#else
	float output_sigma = sqrt(float(width*height)) * output_sigma_factor/m_cellSize;
#endif

	float left = -0.5/(output_sigma*output_sigma);

	int radius_w = int(workRegionWidth/2);
	int radius_h = int(workRegionHeight/2);

	for( int y = 0; y < workRegionHeight; y++ ) {
		int index =  y*workRegionWidth;

		float ry = y-radius_h;

		for( int x = 0; x < workRegionWidth; x++ ) {
			float rx = x-radius_w;
			if (abs(rx)<0.001 && abs(ry)<0.001)
			{
				int terminal = 0;
			}
			gaussianWeight[index++] = exp(left * (ry * ry + rx * rx));
		}
	}
	float *tmpGuassianWeight = new float[workRegionHeight*workRegionWidth];
	memset(tmpGuassianWeight,0,sizeof(float)*workRegionWidth*workRegionHeight);
	circshift(gaussianWeight,tmpGuassianWeight,workRegionWidth,workRegionHeight,workRegionWidth,workRegionHeight);

//	memcpy(gaussianWeightDFT,tmpGuassianWeight,sizeof(float)*workRegionHeight*workRegionWidth);
//	fft->realForwardFull(gaussianWeightDFT);
/*
 	for (int km=0; km<workRegionHeight*workRegionWidth; km++)
	{
		gaussianWeightDFT[km*2] = tmpGuassianWeight[km];
		gaussianWeightDFT[km*2+1] = 0;
	}
*/
	cu_compFFT_forward(tmpGuassianWeight,gaussianWeightDFT, workRegionWidth, workRegionHeight,0,cufft_odata, cufft_plan,cufft_rawdata);

	delete []tmpGuassianWeight;
	//fft.forward(gaussianWeight,gaussianWeightDFT);
}

#ifdef FFT
void CirculantTracker::resizeImages( int width,int height ) {
	
	templateNew = new float[width*height];
	memset(templateNew,0,sizeof(float)*width*height);
	
	templateOld = new float[width*height];
	memset(templateOld,0,sizeof(float)*width*height);

#else
void CirculantTracker::resizeImages( int width,int height, int feaDim) {
	
	templateNew = new float[width*height*feaDim];
	memset(templateNew,0,sizeof(float)*width*height*feaDim);
	
	templateOld = new float[width*height*feaDim];
	memset(templateOld,0,sizeof(float)*width*height*feaDim);
#endif
	cosine = new float[width*height];
	memset(cosine,0,sizeof(float)*width*height);
	
	k = new float[width*height];
	memset(k,0,sizeof(float)*width*height);
	
	kf =  new float[width*height*2];
	memset(kf,0,sizeof(float)*width*height*2);
	
	alphaf = new float[width*height*2];
	memset(alphaf,0,sizeof(float)*width*height*2);
	
	newAlphaf = new float[width*height*2];
	memset(newAlphaf,0,sizeof(float)*width*height*2);
	
	response = new float[width*height];
	memset(response,0,sizeof(float)*width*height);
	
	tmpReal0 = new float[width*height];
	memset(tmpReal0,0,sizeof(float)*width*height);
	
	tmpReal1 = new float[width*height];
	memset(tmpReal1,0,sizeof(float)*width*height);
#ifdef FFT
	tmpFourier0 = new float[width*height*2];
	memset(tmpFourier0,0,sizeof(float)*width*height*2);
	
	tmpFourier1 = new float[width*height*2];
	memset(tmpFourier1,0,sizeof(float)*width*height*2);
	
	tmpFourier2 = new float[width*height*2];
	memset(tmpFourier2,0,sizeof(float)*width*height*2);

#else
	tmpFourier0 = new float[width*height*2*feaDim];
	memset(tmpFourier0,0,sizeof(float)*width*height*2*feaDim);

	tmpFourier1 = new float[width*height*2*feaDim];
	memset(tmpFourier1,0,sizeof(float)*width*height*2*feaDim);

	tmpFourier2 = new float[width*height*2*feaDim];
	memset(tmpFourier2,0,sizeof(float)*width*height*2*feaDim);
#endif	
	
	gaussianWeight = new float[width*height];
	memset(gaussianWeight,0,sizeof(float)*width*height);
	
	gaussianWeightDFT = new float[width*height*2];
	memset(gaussianWeightDFT,0,sizeof(float)*width*height*2);
}

void CirculantTracker::performTracking( unsigned char* image ) {
	TC long begin = clock();
	updateTrackLocation(image);
	LOG << "updateTrackLocation  " << clock()-begin<<endl;
	TC begin = clock();
	if( interp_factor != 0 )
		performLearning(image);
	LOG << "performLearning  " << clock()-begin<<endl;
}

void CirculantTracker::multiplyComplex( float *complexA , float *complexB , float *complexC ) {


	for( int y = 0; y < workRegionHeight; y++ ) {

		int indexA = 0 + y*workRegionWidth*2;
		int indexB = 0 + y*workRegionWidth*2;
		int indexC = 0 + y*workRegionWidth*2;

		for( int x = 0; x < workRegionWidth; x++, indexA += 2 , indexB += 2  ,indexC += 2 ) {

			float realA = complexA[indexA];
			float imgA = complexA[indexA+1];
			float realB = complexB[indexB];
			float imgB = complexB[indexB+1];

			complexC[indexC] = realA*realB - imgA*imgB;
			complexC[indexC+1] = realA*imgB + imgA*realB;
		}
	}
}

void CirculantTracker::updateTrackLocation(unsigned char* image) {
	
	TC long t1 = clock();
	get_subwindow(image, templateNew);
	LOG<<"get subwindow time = "<<clock()-t1<<endl;
	// calculate response of the classifier at all locations
	// matlab: k = dense_gauss_kernel(sigma, x, z);
	
	TC t1 = clock();
	dense_gauss_kernel(sigma, templateNew, templateOld,k);
	LOG<<"dense_gauss_kernel time = "<<clock()-t1<<endl;

	/*fft.forward(k,kf);*/
	TC t1 = clock();

//	memcpy(kf,k,sizeof(float)*workRegionHeight*workRegionWidth);
//	fft->realForwardFull(kf);
/*
		for (int km=0; km<workRegionHeight*workRegionWidth; km++)
		{
			kf[km*2] = k[km];
			kf[km*2+1] = 0;
		}
*/
		cu_compFFT_forward(k,kf, workRegionWidth, workRegionHeight,0, cufft_odata, cufft_plan,cufft_rawdata);



//	float* return_data = new float[workRegionHeight*workRegionWidth*2];

//	cu_compFFT_forward(k, workRegionWidth, workRegionHeight, return_data);

//	kf = return_data;

	// response = real(ifft2(alphaf .* fft2(k)));   %(Eq. 9)
	LOG<<"memcpy time  = "<<clock()-t1<<endl;

	TC t1 = clock();
	multiplyComplex(alphaf, kf, tmpFourier0);
	//fft->inverse(tmpFourier0, response);
TC	long qq = clock();
	float *tmp_response = new float[workRegionHeight*workRegionWidth*2];
//	float *tmp_response2 = new float[workRegionHeight*workRegionWidth*2];
	memcpy(tmp_response,tmpFourier0,sizeof(float)*workRegionHeight*workRegionWidth*2);
//	memcpy(tmp_response2,tmpFourier0,sizeof(float)*workRegionHeight*workRegionWidth*2);
//	fft->complexInverse(tmp_response, true);
	cu_compFFT_forward(tmp_response,tmp_response, workRegionWidth, workRegionHeight,1,cufft_odata,cufft_plan,cufft_rawdata);
/*

		for (int i = 0; i < workRegionHeight*workRegionWidth*2; i++)	
		{
			LOG<<"first inverse------------------------"<<i<<"  "<< tmp_response[i] <<"  "<<tmp_response2[i]<<endl;
		}
*/

	LOG<< "Inverse---------------------------------------"<<clock()-qq<<endl;



	int N = workRegionWidth*workRegionHeight;
	
	
	for( int i = 0; i < N; i++ ) {
		response[i] = tmp_response[i*2];
	}
	delete []tmp_response;

	// find the pixel with the largest response
	int indexBest = -1;
	double valueBest = -1;
	for( int i = 0; i < N; i++ ) {
		double v = response[i];
		if( v > valueBest ) {
			valueBest = v;
			indexBest = i;
		}
	}

	int peakX = indexBest % workRegionWidth;
	int peakY = indexBest / workRegionHeight;

	if (peakY>(workRegionHeight/2-1))
	{
		peakY = peakY-workRegionHeight;
	}
	if (peakX>(workRegionWidth/2-1))
	{
		peakX = peakX-workRegionWidth;
	}

	//if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
	//	vert_delta = vert_delta - size(zf,1);
	//end
	//if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
	//		horiz_delta = horiz_delta - size(zf,2);
	//end

	//// sub-pixel peak estimation
	//subpixelPeak(peakX, peakY);

	//// peak in region's coordinate system
	//float deltaX = (peakX+offX) - workRegionWidth/2;
	//float deltaY = (peakY+offY) - workRegionHeight/2;

	// convert peak location into image coordinate system
	//regionTrack.m_xMin = regionTrack.m_xMin + peakX**m_cellSize;
#ifdef FFT
		regionTrack.m_xMin = regionTrack.m_xMin + peakX*stepX;
	regionTrack.m_yMin = regionTrack.m_yMin + peakY*stepY;
#else
	regionTrack.m_xMin = regionTrack.m_xMin + peakX*m_cellSize;
	regionTrack.m_yMin = regionTrack.m_yMin + peakY*m_cellSize;
#endif


	updateRegionOut();
	LOG<<"updata last program time =  "<<clock()-t1<<endl;
}

/**
	 * Refine the local-peak using a search algorithm for sub-pixel accuracy.
	 */
void CirculantTracker::subpixelPeak(int peakX, int peakY) {
		// this function for r was determined empirically by using work regions of 32,64,128
		int r = min(2,workRegionWidth/25);
		if( r < 0 )
			return;

		/*localPeak.setSearchRadius(r);
		localPeak.search(peakX,peakY);

		offX = localPeak.getPeakX() - peakX;
		offY = localPeak.getPeakY() - peakY;*/
	}


void CirculantTracker::updateRegionOut() {
	regionOut.m_xMin = (regionTrack.m_xMin+((int)regionTrack.m_width)/2)-((int)regionOut.m_width)/2;
	regionOut.m_yMin = (regionTrack.m_yMin+((int)regionTrack.m_height)/2)-((int)regionOut.m_height)/2;
}


/**
	 * Update the alphas and the track's appearance
	 */
void CirculantTracker::performLearning(unsigned char* image) {

	long begin,end,time;
	TC begin = clock();

		// use the update track location
		get_subwindow(image, templateNew);
		t_subwindow += clock()-begin;
		LOG << " subwindow  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();

		// Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
		//	k = dense_gauss_kernel(sigma, x);
		dense_gauss_kernel(sigma, templateNew, templateNew, k);
		TC t_dense_gause += clock()-begin;
		LOG << " dense_gauss_kernel  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();

		/*fft.forward(k,kf);*/
//		memcpy(kf,k,sizeof(float)*workRegionHeight*workRegionWidth);

//		fft->realForwardFull(kf);
 /*	for (int km=0; km<workRegionHeight*workRegionWidth; km++)
	{
		kf[km*2] = k[km];
		kf[km*2+1] = 0;
	}
*/
	cu_compFFT_forward(k,kf, workRegionWidth, workRegionHeight,0,cufft_odata,cufft_plan,cufft_rawdata);

/*		for (int km=0; km<workRegionHeight*workRegionWidth; km++)
		{
			kf[km*2] = k[km];
			kf[km*2+1] = 0;
		}

		cu_compFFT_forward(kf, workRegionWidth, workRegionHeight);
*/
		TC t_fft += clock()-begin;
		LOG << " realForwardFull  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();


		// new_alphaf = yf ./ (fft2(k) + lambda);   %(Eq. 7)
		computeAlphas(gaussianWeightDFT, kf, lambda, newAlphaf,workRegionWidth,workRegionHeight);
		LOG << " computeAlphas  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();


		// subsequent frames, interpolate model
		// alphaf = (1 - interp_factor) * alphaf + interp_factor * new_alphaf;
		int N = workRegionWidth*workRegionHeight*2;
		for( int i = 0; i < N; i++ ) {
			alphaf[i] = (1-interp_factor)*alphaf[i] + interp_factor*newAlphaf[i];
		}
		TC t_hog += clock()-begin;
		LOG << " for_loop1  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();


		// Set the previous image to be an interpolated version
		//		z = (1 - interp_factor) * z + interp_factor * new_z;

#ifdef FFT
		N = workRegionWidth* workRegionHeight;
#else
		N = workRegionWidth* workRegionHeight*9;
#endif
		for( int i = 0; i < N; i++ ) {
			templateOld[i] = (1-interp_factor)* templateOld[i] + interp_factor*templateNew[i];
		}
		LOG << " for_loop2  " << clock()-begin << "clocks"<<endl;
		TC begin = clock();
	}


void CirculantTracker::dense_gauss_kernel( double sigma , float *x , float *y , float *k ) {

		float *xf=tmpFourier0;
		float *yf = tmpFourier2;
		float *xyf=tmpFourier2;
		float *xy = tmpReal0;
		double yy;

		// find x in Fourier domain
		/*fft.forward(x, xf);*/
	TC long t1 = clock();
#ifdef FFT
//		memcpy(xf,x,sizeof(float)*workRegionHeight*workRegionWidth);
//		fft->realForwardFull(xf);
/*
 	for (int km=0; km<workRegionHeight*workRegionWidth; km++)
	{
		xf[km*2] = x[km];
		xf[km*2+1] = 0;
	}
*/
//	LOG<< "values x to xf is "<< clock()-t1<<endl;
	cu_compFFT_forward(x,xf, workRegionWidth, workRegionHeight,0,cufft_odata,cufft_plan,cufft_rawdata);
	LOG<<"cf_fft forward is "<<clock()-t1<<endl;
		LOG<<"-------------------"<<endl;
#else
		for (int z=0;z<9;z++)
		{
			float *start_x = x+workRegionHeight*workRegionWidth*z;
			float *start_xf = xf+workRegionHeight*workRegionWidth*2*z;
			memcpy(start_xf,start_x,sizeof(float)*workRegionHeight*workRegionWidth);
			fft->realForwardFull(start_xf);
		}
#endif
		TC t1 = clock();
		float xx = imageDotProduct(x,workRegionWidth,workRegionHeight);
		LOG<<"imageDot is "<<clock()-t1<<endl;

		if( x != y ) {
			// general case, x and y are different
			yf = tmpFourier1;
			/*fft.forward(y,yf);*/
#ifdef FFT
	//		memcpy(yf,y,sizeof(float)*workRegionHeight*workRegionWidth);
	//		fft->realForwardFull(yf)

	TC t1 = clock();
/*
 	for (int km=0; km<workRegionHeight*workRegionWidth; km++)
	{
		yf[km*2] = y[km];
		yf[km*2+1] = 0;
	}
*/
	cu_compFFT_forward(y,yf, workRegionWidth, workRegionHeight,0,cufft_odata,cufft_plan,cufft_rawdata);
	LOG<<" fft2 is "<<clock()-t1 <<endl;
#else
			for (int z=0;z<9;z++)
			{
				float *start_y = y+workRegionHeight*workRegionWidth*z;
				float *start_yf = yf+workRegionHeight*workRegionWidth*2*z;
				memcpy(start_yf,start_y,sizeof(float)*workRegionHeight*workRegionWidth);
				fft->realForwardFull(start_yf);
			}
#endif
			yy = imageDotProduct(y,workRegionWidth,workRegionHeight);
		} else {
			// auto-correlation of x, avoid repeating a few operations
			yf = xf;
			yy = xx;
		}

		//----   xy = invF[ F(x)*F(y) ]
		// cross-correlation term in Fourier domain

	TC	t1 = clock();
		elementMultConjB(xf,yf,xyf,workRegionWidth,workRegionHeight);
		LOG<< "lementMulti is "<< clock()-t1 <<endl;
		// convert to spatial domain
		/*fft.inverse(xyf,xy);*/
#ifdef FFT
	TC	t1 = clock();
		float *tmp_xyf = new float[workRegionHeight*workRegionWidth*2];
//		float *tmp_xyf2 = new float[workRegionHeight*workRegionWidth*2];
		memcpy(tmp_xyf,xyf,sizeof(float)*workRegionHeight*workRegionWidth*2);
		fft->complexInverse(tmp_xyf, true);
//		cu_compFFT_forward(tmp_xyf,tmp_xyf, workRegionWidth, workRegionHeight, 1, cufft_odata,cufft_plan);
//		memcpy(tmp_xyf2,xyf,sizeof(float)*workRegionHeight*workRegionWidth*2);
/*
		LOG<<"------------------------"<<count_fft++<<endl;
		if(x!=y)
		{
			fft->complexInverse(tmp_xyf, true);
			LOG<<"x======================y"<<endl;
		}
		else
		{
			cu_compFFT_forward(tmp_xyf, workRegionWidth, workRegionHeight, 1, cufft_odata,cufft_plan);
			LOG<<"x!!!!!!!!!!!!!!!!!!!!!!!!y"<<endl;
		}
*/
/*
		for (int i = 0; i < workRegionHeight*workRegionWidth*2; i++)	
		{
			LOG<<"second inverse----------------------"<<i<<" "<< tmp_xyf[i] <<"  "<<tmp_xyf2[i]<<endl;
		}
*/
		LOG<< " Inverse in dense is---------------------------------------------- "<<clock()-t1<<endl;
/*
	long qq = clock();
	float *tmp_response = new float[workRegionHeight*workRegionWidth*2];
	memcpy(tmp_response,tmpFourier0,sizeof(float)*workRegionHeight*workRegionWidth*2);
	//fft->complexInverse(tmp_response, true);
	cu_compFFT_forward(tmp_response, workRegionWidth, workRegionHeight,1,cufft_odata,cufft_plan);
	LOG<< "Inverse---------------------------------------"<<clock()-qq<<endl;

*/



//
#else
		float *tmp_xyf = new float[workRegionHeight*workRegionWidth*2*9];

		for (int z=0;z<9;z++)
		{
			float *start_tmp_xyf = tmp_xyf+workRegionHeight*workRegionWidth*2*z;
			float *start_xyf = xyf+workRegionHeight*workRegionWidth*2*z;
			memcpy(start_tmp_xyf,start_xyf,sizeof(float)*workRegionHeight*workRegionWidth*2);
			//fft->realForwardFull(start_tmp_xyf);
			fft->complexInverse(start_tmp_xyf, true);
		}
#endif

#ifdef FFT
		int N = workRegionWidth*workRegionHeight;

		for( int i = 0; i < N; i++ ) {
			xy[i] = tmp_xyf[i*2];
		}
#else
		for (int y=0;y<workRegionHeight;y++)
		{
			for (int x=0;x<workRegionWidth;x++)
			{
				float fSum = 0;
				for (int z=0;z<9;z++)
				{
					fSum+=tmp_xyf[z*workRegionHeight*workRegionWidth*2+y*workRegionWidth*2+x*2];
				}
				xy[y*workRegionWidth+x] = fSum;
			}
		}
#endif
		delete []tmp_xyf;
		//circshift(xy,tmpReal1,workRegionWidth,workRegionHeight,workRegionWidth,workRegionHeight);

		// calculate gaussian response for all positions
		
	TC	t1 = clock();
		cu_gaussianKernel(xx, yy, xy, sigma, k,workRegionWidth,workRegionHeight);
		LOG<<"gauseKernel is "<< clock()-t1 <<endl;
	}

void CirculantTracker::circshift( float *a, float *b, int aWidth, int aHeight, int bWidth, int bHeight) {
	int w2 = aWidth/2;
	int h2 = bHeight/2;

	for( int y = 0; y < aHeight; y++ ) {
		int yy = (y+h2)%aHeight;

		for( int x = 0; x < aWidth; x++ ) {
			int xx = (x+w2)%aWidth;

			b[xx+yy*bWidth] = a[x+y*aWidth];
		}
	}
}

	/**
	 * Computes the dot product of the image with itself
	 */
float CirculantTracker::imageDotProduct(float *a, int nWidth, int nHeight) {

	double total = 0;
#ifdef FFT
	int N = nWidth*nHeight;
#else
	int N = nWidth*nHeight*9;
#endif
	for( int index = 0; index < N; index++ ) {
		float value = a[index];
		total += value*value;
	}
	return total;
}

	/**
	 * Element-wise multiplication of 'a' and the complex conjugate of 'b'
	 */
void CirculantTracker::elementMultConjB( float *a , float *b , float *output, int nWidth, int nHeight ) {

#ifndef FFT
	for (int z=0;z<9;z++)
	{
#endif
		for( int y = 0; y < nHeight; y++ ) {
#ifdef FFT
			int index =  y*nWidth*2;
#else
			int index =  z*nHeight*nWidth*2+y*nWidth*2;
#endif
			for( int x = 0; x < nWidth; x++, index += 2 ) {

				float realA = a[index];
				float imgA = a[index+1];
				float realB = b[index];
				float imgB = b[index+1];

				output[index] = realA*realB + imgA*imgB;
				output[index+1] = -realA*imgB + imgA*realB;
			}
		}
#ifndef FFT
	}
#endif
	}

	/**
	 * new_alphaf = yf ./ (fft2(k) + lambda);   %(Eq. 7)
	 */
void CirculantTracker::computeAlphas( float *yf , float *kf , float lambda ,
										 float *alphaf, int nWidth, int nHeight ) {

		for( int y = 0; y < nHeight; y++ ) {

			int index = y*nWidth*2;

			for( int x = 0; x < nWidth; x++, index += 2 ) {
				double a = yf[index];
				double b = yf[index+1];

				double c = kf[index] + lambda;
				double d = kf[index+1];

				double bottom = c*c + d*d;

				alphaf[index] = (a*c + b*d)/bottom;
				alphaf[index+1] = (b*c - a*d)/bottom;
			}
		}
	}

	/**
	 * Computes the output of the Gaussian kernel for each element in the target region
	 *
	 * k = exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(x)));
	 *
	 * @param xx ||x||^2
	 * @param yy ||y||^2
	 */
void CirculantTracker::gaussianKernel( float xx , float yy , float *xy , double sigma  , float *output, int nWidth, int nHeigth ) {
		float sigma2 = sigma*sigma;

#ifdef FFT
		float N = nWidth*nHeigth;
#else
		float N = nWidth*nHeigth*9;
#endif
		for( int index = 0; index < nHeigth*nWidth; index++ ) {

//			for( int x = 0; x < nWidth; x++ ) {

//				int index =  y*nWidth +x;
				// (xx + yy - 2 * xy) / numel(x)
				float value = (xx + yy - 2*xy[index])/N;

				float v = exp(-max(0.0, value) / sigma2);

				output[index] = v;
//			}
		}
	}

	/**
	 * Copies the target into the output image and applies the cosine window to it.
	 */

#ifdef FFT
void CirculantTracker::get_subwindow( unsigned char* image , float *output ) {
	//IplImage *img1 = cvCreateImage(cvSize(1920,1080), 8, 1); 
	//for (int i=0; i<1080; i++)
	//{
	//	for (int j=0; j<1920; j++)
	//	{
	//		
	//		((uchar *)(img1->imageData + i*img1->widthStep))[j] = image[i*1920+j];
	//	}
	//}
	//cvSaveImage("debug_im_2.jpg",img1);
	int index = 0;
	//IplImage *img = cvCreateImage(cvSize(workRegionWidth,workRegionHeight), 8, 1); 
	for (int y=0; y<workRegionHeight; y++)
	{
		int yy = regionTrack.m_yMin + y;
		for (int x=0; x<workRegionWidth; x++)
		{
			int xx = regionTrack.m_xMin + x;
			if (xx<0||xx>=m_imWidth||yy<0||yy>=m_imHeight)
			{
				output[index++] = 0;
			}
			else
			{
				//float onePixel = image[yy*m_imWidth+xx];
				//((uchar *)(img->imageData + y*img->widthStep))[x] = image[yy*m_imWidth+xx];
				output[index++] = (float(image[yy*m_imWidth+xx])/255-0.5)*cosine[y*workRegionWidth+x];
			}	
		}
	}
	//cvSaveImage("debug_im.jpg",img);
		// copy the target region


	}
#else
void CirculantTracker::get_subwindow( unsigned char* image , float *output ) {

	int track_w = regionTrack.m_width;
	int track_h = regionTrack.m_height;
	float *tmp_data = new float[track_w*track_h];
	for (int x=0;x<track_w;x++)
	{
		int xx = regionTrack.m_xMin + x;
		for (int y=0;y<track_h;y++)
		{
			int yy = regionTrack.m_yMin + y;
			if (xx<0||xx>=m_imWidth||yy<0||yy>=m_imHeight)
			{
				tmp_data[x*track_h+y] = 0.0;
			}
			else
			{
				tmp_data[x*track_h+y] = float(image[yy*m_imWidth+xx]);
			}
		}
	}

	float *cu_data;


	cudaError_t cudaStatus = cudaMalloc(&cu_data,track_w*track_h*sizeof(float));

	cudaMemcpy(cu_data, tmp_data, track_w*track_h * sizeof(float), cudaMemcpyHostToDevice);

	TC long begin = clock();

	float* cu_hogFea = cu_hog( cu_data, track_h, track_w, 1, m_cellSize, 9, 10 );
	LOG<<" hogFea-----------------------   "<< clock()-begin<<endl;
	float *hogFea = new float[9*workRegionHeight*workRegionWidth];
	cudaMemcpy(hogFea, cu_hogFea, 9*workRegionHeight*workRegionWidth*sizeof(float),cudaMemcpyDeviceToHost);


//	double *hogFea = hog(tmp_data,track_h, track_w, 1, m_cellSize, 9, 10,cu_data );
	for (int z=0;z<9;z++)
	{
		for (int y=0;y<workRegionHeight;y++)
		{
			for (int x=0;x<workRegionWidth;x++)
			{
				output[z*workRegionHeight*workRegionWidth+y*workRegionWidth+x] = (float(hogFea[z*workRegionHeight*workRegionWidth+x*workRegionHeight+y]))*cosine[y*workRegionWidth+x];
			}
		}
	}

	delete []tmp_data;

	cudaFree(cu_hogFea);
	cudaFree(cu_data);


	delete []hogFea;
//	delete []hogFea;

	//int index = 0;
	////IplImage *img = cvCreateImage(cvSize(workRegionWidth,workRegionHeight), 8, 1); 
	//for (int y=0; y<workRegionHeight; y++)
	//{
	//	int yy = regionTrack.m_yMin + y;
	//	for (int x=0; x<workRegionWidth; x++)
	//	{
	//		int xx = regionTrack.m_xMin + x;
	//		if (xx<0||xx>=m_imWidth||yy<0||yy>=m_imHeight)
	//		{
	//			output[index++] = 0;
	//		}
	//		else
	//		{
	//			//float onePixel = image[yy*m_imWidth+xx];
	//			//((uchar *)(img->imageData + y*img->widthStep))[x] = image[yy*m_imWidth+xx];
	//			output[index++] = (float(image[yy*m_imWidth+xx])/255-0.5)*cosine[y*workRegionWidth+x];
	//		}	
	//	}
	//}


	}
#endif
	/**
	 * The location of the target in the image
	 */
FloatRect CirculantTracker::getTargetLocation() {
		return regionOut;
	}
