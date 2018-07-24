#include "GeneralFFT_2D.h"
#include <iostream>


GeneralFFT_2D::GeneralFFT_2D(void)
{
	t = NULL;
	fftColumns = NULL;
	fftRows = NULL;
	temp = NULL;
	temp2 = NULL;
}


GeneralFFT_2D::~GeneralFFT_2D(void)
{
	if (t!=NULL)
	{
		delete []t;
		t = NULL;
	}
	if (fftColumns!=NULL)
	{
		delete fftColumns;
		fftColumns = NULL;
	}
	if (fftRows!=NULL)
	{
		delete fftRows;
		fftRows = NULL;
	}
	if (temp!=NULL)
	{
		delete []temp;
		temp = NULL;
	}
	if (temp2!=NULL)
	{
		int n2d2 = columns / 2 + 1;
		for (int i=0;i<n2d2;i++)
		{
			delete [] temp2[i];
		}
		delete []temp2;
	}
}

/**
	 * Creates new instance of DoubleFFT_2D.
	 *
	 * @param rows
	 *            number of rows
	 * @param columns
	 *            number of columns
	 */
GeneralFFT_2D::GeneralFFT_2D(int rows_i, int columns_i) {
		t = NULL;
		fftColumns = NULL;
		fftRows = NULL;
		temp = NULL;
		temp2 = NULL;
	    
		rows = rows_i;
		columns = columns_i;
		isPowerOfTwo = false;

			isPowerOfTwo = true;

			int oldNthreads = 1;
			int nt = 8 * oldNthreads * rows;
			if (2 * columns == 4 * oldNthreads) {
				nt >>= 1;
			} else if (2 * columns < 4 * oldNthreads) {
				nt >>= 2;
			}
			t = new float[nt];
			memset(t,0,sizeof(float)*(nt));

		fftRows = new GeneralFFT_1D(rows);
		fftColumns = new GeneralFFT_1D(columns);

		temp = new float[2 * rows];
		memset(temp,0,sizeof(float)*(2 * rows));
		temp2 = NULL;
	}

	/**
	 * Computes 2D inverse DFT of complex data leaving the result in
	 * <code>a</code>. The data is stored in 1D array in row-major order.
	 * Complex number is stored as two float values in sequence: the real and
	 * imaginary part, i.e. the input array must be of size rows*2*columns. The
	 * physical layout of the input data has to be as follows:<br>
	 *
	 * <pre>
	 * a[k1*2*columns+2*k2] = Re[k1][k2],
	 * a[k1*2*columns+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
	 * </pre>
	 *
	 * @param a
	 *            data to transform
	 * @param scale
	 *            if true then scaling is performed
	 *
	 */
void GeneralFFT_2D::complexInverse(float* a, bool scale) {
		// handle special case
		if( rows == 1 || columns == 1 ) {
			if( rows > 1 )
				fftRows->complexInverse(a, scale);
			else
				fftColumns->complexInverse(a, scale);
			return;
		}


			int oldn2 = columns;
			columns = 2 * columns;
			for (int r = 0; r < rows; r++) {
				fftColumns->complexInverse(a, r * columns, scale);
			}
			cdft2d_sub(1, a, scale);
			columns = oldn2;

	}



	/**
	 * Computes 2D forward DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real forward transform, i.e. you will get the
	 * same result as from <code>complexForward</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size rows*2*columns, with only the first rows*columns
	 * elements filled with real data. To get back the original data, use
	 * <code>complexInverse</code> on the output of this method.
	 *
	 * @param a
	 *            data to transform
	 */
void GeneralFFT_2D::realForwardFull(float* a) {
		// handle special case
		if( rows == 1 || columns == 1 ) {
			if( rows > 1 )
				fftRows->realForwardFull(a);
			else
				fftColumns->realForwardFull(a);
			return;
		}


		for (int r = 0; r < rows; r++) {
			fftColumns->realForward(a, r * columns);
		}
		cdft2d_sub(-1, a, true);
		rdft2d_sub(1, a);
		fillSymmetric(a);
	}


	/**
	 * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
	 * . This method computes full real inverse transform, i.e. you will get the
	 * same result as from <code>complexInverse</code> called with all imaginary
	 * part equal 0. Because the result is stored in <code>a</code>, the input
	 * array must be of size rows*2*columns, with only the first rows*columns
	 * elements filled with real data.
	 *
	 * @param a
	 *            data to transform
	 *
	 * @param scale
	 *            if true then scaling is performed
	 */


void GeneralFFT_2D::rdft2d_sub(int isgn, float* a) {
		int n1h, j;
		float xi;
		int idx1, idx2;

		n1h = rows >> 1;
		if (isgn < 0) {
			for (int i = 1; i < n1h; i++) {
				j = rows - i;
				idx1 = i * columns;
				idx2 = j * columns;
				xi = a[idx1] - a[idx2];
				a[idx1] += a[idx2];
				a[idx2] = xi;
				xi = a[idx2 + 1] - a[idx1 + 1];
				a[idx1 + 1] += a[idx2 + 1];
				a[idx2 + 1] = xi;
			}
		} else {
			for (int i = 1; i < n1h; i++) {
				j = rows - i;
				idx1 = i * columns;
				idx2 = j * columns;
				a[idx2] = 0.5f * (a[idx1] - a[idx2]);
				a[idx1] -= a[idx2];
				a[idx2 + 1] = 0.5f * (a[idx1 + 1] + a[idx2 + 1]);
				a[idx1 + 1] -= a[idx2 + 1];
			}
		}
	}

void GeneralFFT_2D::cdft2d_sub(int isgn, float* a, bool scale) {
		int idx1, idx2, idx3, idx4, idx5;
		if (isgn == -1) {
			if (columns > 4) {
				for (int c = 0; c < columns; c += 8) {
					for (int r = 0; r < rows; r++) {
						idx1 = r * columns + c;
						idx2 = 2 * r;
						idx3 = 2 * rows + 2 * r;
						idx4 = idx3 + 2 * rows;
						idx5 = idx4 + 2 * rows;
						t[idx2] = a[idx1];
						t[idx2 + 1] = a[idx1 + 1];
						t[idx3] = a[idx1 + 2];
						t[idx3 + 1] = a[idx1 + 3];
						t[idx4] = a[idx1 + 4];
						t[idx4 + 1] = a[idx1 + 5];
						t[idx5] = a[idx1 + 6];
						t[idx5 + 1] = a[idx1 + 7];
					}
					fftRows->complexForward(t, 0);
					fftRows->complexForward(t, 2 * rows);
					fftRows->complexForward(t, 4 * rows);
					fftRows->complexForward(t, 6 * rows);
					for (int r = 0; r < rows; r++) {
						idx1 = r * columns + c;
						idx2 = 2 * r;
						idx3 = 2 * rows + 2 * r;
						idx4 = idx3 + 2 * rows;
						idx5 = idx4 + 2 * rows;
						a[idx1] = t[idx2];
						a[idx1 + 1] = t[idx2 + 1];
						a[idx1 + 2] = t[idx3];
						a[idx1 + 3] = t[idx3 + 1];
						a[idx1 + 4] = t[idx4];
						a[idx1 + 5] = t[idx4 + 1];
						a[idx1 + 6] = t[idx5];
						a[idx1 + 7] = t[idx5 + 1];
					}
				}
			} else if (columns == 4) {
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					idx3 = 2 * rows + 2 * r;
					t[idx2] = a[idx1];
					t[idx2 + 1] = a[idx1 + 1];
					t[idx3] = a[idx1 + 2];
					t[idx3 + 1] = a[idx1 + 3];
				}
				fftRows->complexForward(t, 0);
				fftRows->complexForward(t, 2 * rows);
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					idx3 = 2 * rows + 2 * r;
					a[idx1] = t[idx2];
					a[idx1 + 1] = t[idx2 + 1];
					a[idx1 + 2] = t[idx3];
					a[idx1 + 3] = t[idx3 + 1];
				}
			} else if (columns == 2) {
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					t[idx2] = a[idx1];
					t[idx2 + 1] = a[idx1 + 1];
				}
				fftRows->complexForward(t, 0);
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					a[idx1] = t[idx2];
					a[idx1 + 1] = t[idx2 + 1];
				}
			}
		} else {
			if (columns > 4) {
				for (int c = 0; c < columns; c += 8) {
					for (int r = 0; r < rows; r++) {
						idx1 = r * columns + c;
						idx2 = 2 * r;
						idx3 = 2 * rows + 2 * r;
						idx4 = idx3 + 2 * rows;
						idx5 = idx4 + 2 * rows;
						t[idx2] = a[idx1];
						t[idx2 + 1] = a[idx1 + 1];
						t[idx3] = a[idx1 + 2];
						t[idx3 + 1] = a[idx1 + 3];
						t[idx4] = a[idx1 + 4];
						t[idx4 + 1] = a[idx1 + 5];
						t[idx5] = a[idx1 + 6];
						t[idx5 + 1] = a[idx1 + 7];
					}
					fftRows->complexInverse(t, 0, scale);
					fftRows->complexInverse(t, 2 * rows, scale);
					fftRows->complexInverse(t, 4 * rows, scale);
					fftRows->complexInverse(t, 6 * rows, scale);
					for (int r = 0; r < rows; r++) {
						idx1 = r * columns + c;
						idx2 = 2 * r;
						idx3 = 2 * rows + 2 * r;
						idx4 = idx3 + 2 * rows;
						idx5 = idx4 + 2 * rows;
						a[idx1] = t[idx2];
						a[idx1 + 1] = t[idx2 + 1];
						a[idx1 + 2] = t[idx3];
						a[idx1 + 3] = t[idx3 + 1];
						a[idx1 + 4] = t[idx4];
						a[idx1 + 5] = t[idx4 + 1];
						a[idx1 + 6] = t[idx5];
						a[idx1 + 7] = t[idx5 + 1];
					}
				}
			} else if (columns == 4) {
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					idx3 = 2 * rows + 2 * r;
					t[idx2] = a[idx1];
					t[idx2 + 1] = a[idx1 + 1];
					t[idx3] = a[idx1 + 2];
					t[idx3 + 1] = a[idx1 + 3];
				}
				fftRows->complexInverse(t, 0, scale);
				fftRows->complexInverse(t, 2 * rows, scale);
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					idx3 = 2 * rows + 2 * r;
					a[idx1] = t[idx2];
					a[idx1 + 1] = t[idx2 + 1];
					a[idx1 + 2] = t[idx3];
					a[idx1 + 3] = t[idx3 + 1];
				}
			} else if (columns == 2) {
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					t[idx2] = a[idx1];
					t[idx2 + 1] = a[idx1 + 1];
				}
				fftRows->complexInverse(t, 0, scale);
				for (int r = 0; r < rows; r++) {
					idx1 = r * columns;
					idx2 = 2 * r;
					a[idx1] = t[idx2];
					a[idx1 + 1] = t[idx2 + 1];
				}
			}
		}
	}


void GeneralFFT_2D::fillSymmetric( float* a) {
		int twon2 = 2 * columns;
		int idx1, idx2, idx3, idx4;
		int n1d2 = rows / 2;

		for (int r = (rows - 1); r >= 1; r--) {
			idx1 = r * columns;
			idx2 = 2 * idx1;
			for (int c = 0; c < columns; c += 2) {
				a[idx2 + c] = a[idx1 + c];
				a[idx1 + c] = 0;
				a[idx2 + c + 1] = a[idx1 + c + 1];
				a[idx1 + c + 1] = 0;
			}
		}
		for (int r = 1; r < n1d2; r++) {
			idx2 = r * twon2;
			idx3 = (rows - r) * twon2;
			a[idx2 + columns] = a[idx3 + 1];
			a[idx2 + columns + 1] = -a[idx3];
		}

		for (int r = 1; r < n1d2; r++) {
			idx2 = r * twon2;
			idx3 = (rows - r + 1) * twon2;
			for (int c = columns + 2; c < twon2; c += 2) {
				a[idx2 + c] = a[idx3 - c];
				a[idx2 + c + 1] = -a[idx3 - c + 1];

			}
		}
		for (int r = 0; r <= rows / 2; r++) {
			idx1 = r * twon2;
			idx4 = ((rows - r) % rows) * twon2;
			for (int c = 0; c < twon2; c += 2) {
				idx2 = idx1 + c;
				idx3 = idx4 + (twon2 - c) % twon2;
				a[idx3] = a[idx2];
				a[idx3 + 1] = -a[idx2 + 1];
			}
		}
		a[columns] = -a[1];
		a[1] = 0;
		idx1 = n1d2 * twon2;
		a[idx1 + columns] = -a[idx1 + 1];
		a[idx1 + 1] = 0;
		a[idx1 + columns + 1] = 0;
}



