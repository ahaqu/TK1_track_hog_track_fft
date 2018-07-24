#pragma once
#include "struct.h"
#include "GeneralFFT_2D.h"

#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda.h" 
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cuda_runtime.h"
#include "/usr/local/cuda-6.0/targets/armv7-linux-gnueabihf/include/cufft.h"
//#define FFT

class CirculantTracker
{
public:
	CirculantTracker(void);
	~CirculantTracker(void);
	CirculantTracker(float output_sigma_factor_p, float sigma_p, float lambda_p, float interp_factor_p,
		float padding_p ,
		float maxPixelValue_p);
#ifdef FFT
	void initialize( unsigned char* image , int imWidth, int imHeight, int x0 , int y0 , int regionWidth , int regionHeight );
	void resizeImages( int width,int height);
#else
	void initialize( unsigned char* image , int imWidth, int imHeight, int x0 , int y0 , int regionWidth , int regionHeight, int cellSize );
	void resizeImages( int width,int height, int feaDim);
#endif
	void initialLearning( unsigned char* image,int imWidth, int imHeight );
	void computeCosineWindow( float* cosine,int nWidth, int nHeight );
	void computeGaussianWeights( int width,int height);
	void performTracking( unsigned char* image );
	void updateTrackLocation(unsigned char* image);
	void subpixelPeak(int peakX, int peakY);
	void updateRegionOut();
	void performLearning(unsigned char* image);
	void dense_gauss_kernel( double sigma , float *x , float *y , float *k ) ;
	void circshift( float *a, float *b, int aWidth, int aHeight, int bWidth, int bHeight);
	float imageDotProduct(float *a, int nWidth, int nHeight);
	void elementMultConjB( float *a , float *b , float *output, int nWidth, int nHeight );
	void computeAlphas( float *yf , float *kf , float lambda ,
		float *alphaf, int nWidth, int nHeight );
	void gaussianKernel( float xx , float yy , float *xy , double sigma  , float *output, int nWidth, int nHeigth );
	void get_subwindow( unsigned char* image , float *output );
	FloatRect getTargetLocation() ;
	void multiplyComplex( float *complexA , float *complexB , float *complexC );


public:
	// spatial bandwidth (proportional to target)
	float output_sigma_factor;
	// gaussian kernel bandwidth
	float sigma;
	// regularization term
	float lambda;
	// linear interpolation term.  Adjusts how fast it can learn
	float interp_factor;

	// the maximum pixel value
	float maxPixelValue;

	// extra padding around the selected region
	float padding;

	//// storage for subimage of input image
	float *templateNew;
	//// storage for the subimage of the previous frame
	float *templateOld;

	//// cosine window used to reduce artifacts from FFT
	
	float *cosine;
	//// Storage for the kernel's response

	float *k;

	float *kf;

	//// Learn values.  used to compute weight in linear classifier

	float *alphaf;

	float *newAlphaf;

	//// location of target

	FloatRect regionTrack;

	FloatRect regionOut;
	//// Used for computing the gaussian kernel

	float *gaussianWeight;

	float *gaussianWeightDFT;

	//// detector response

	float *response;

	//// storage for storing temporary results

	float *tmpReal0;

	float *tmpReal1;


	float *tmpFourier0;

	float *tmpFourier1;

	float *tmpFourier2;



	// adjustment from sub-pixel
	float offX,offY;

	// size of the work space in pixels
	int workRegionSize;
	int workRegionWidth;
	int workRegionHeight;
	// conversion from workspace to image pixels
	float stepX,stepY;

	int m_imWidth;
	int m_imHeight;

#ifndef FFT
	int m_cellSize;
#endif
	GeneralFFT_2D *fft;

	long t_fft;
	long t_dense_gause;
	long t_hog;
	long t_subwindow;

	cufftComplex *cufft_odata;
	float *cufft_rawdata;
	cufftHandle cufft_plan;	

};






