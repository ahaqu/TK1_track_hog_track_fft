#pragma once
#include "GeneralFFT_1D.h"
class GeneralFFT_2D
{
public:
	GeneralFFT_2D(void);
	~GeneralFFT_2D(void);
	GeneralFFT_2D(int rows_i, int columns_i);
	void complexInverse(float* a, bool scale);
	void realForwardFull(float* a) ;
	void rdft2d_sub(int isgn, float* a);
	void cdft2d_sub(int isgn, float* a, bool scale);
	void fillSymmetric( float* a);


public:
	int rows;

	int columns;

	float* t;

	GeneralFFT_1D *fftColumns, *fftRows;

	bool isPowerOfTwo ;

	// local storage pre-declared
	float* temp;
	float** temp2;
};

