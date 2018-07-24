#pragma once

//enum Plans {
//	SPLIT_RADIX, MIXED_RADIX, BLUESTEIN
//};
class GeneralFFT_1D
{
public:
	GeneralFFT_1D(void);
	GeneralFFT_1D(int n);
	~GeneralFFT_1D(void);
	void complexForward(float* a);
	void complexForward(float *a, int offa);
	void complexInverse(float *a, bool scale);
	void complexInverse(float* a, int offa, bool scale);
	 void realForward(float* a);
	 void realForward(float* a, int offa) ;
	 void realForwardFull(float* a);
	 void realForwardFull(float* a, int offa);


	 void makewt(int nw);
	 void makeipt(int nw);
	 void makect(int nc, float* c, int startc);
	 
	 void cftfsub(int n, float* a, int offa, int* ip, int nw, float* w);
	 void cftbsub(int n, float* a, int offa, int* ip, int nw, float* w);
	 void bitrv2(int n, int* ip, float* a, int offa);
	 void bitrv2conj(int n, int* ip, float* a, int offa);
	 void bitrv216(float* a, int offa);
	  void bitrv216neg(float* a, int offa);
	  void bitrv208(float* a, int offa);
	  void bitrv208neg(float* a, int offa);
	  void cftf1st(int n, float* a, int offa, float* w, int startw);
	  void cftb1st(int n, float* a, int offa, float* w, int startw);
	  void cftrec4(int n, float* a, int offa, int nw, float* w);
	  int cfttree(int n, int j, int k, float* a, int offa, int nw, float* w) ;
	  void cftleaf(int n, int isplt, float* a, int offa, int nw, float* w);
	  void cftmdl1(int n, float* a, int offa, float* w, int startw);

	  void cftmdl2(int n, float* a, int offa, float* w, int startw);
	  void cftfx41(int n, float* a, int offa, int nw, float* w);
	  void cftf161(float* a, int offa, float* w, int startw);
	  void cftf162(float* a, int offa, float* w, int startw);
	  void cftf081(float* a, int offa, float* w, int startw);
	  void cftx020(float* a, int offa);
	  void cftxb020(float* a, int offa);
	  void cftf082(float* a, int offa, float* w, int startw);
	  void cftb040(float* a, int offa);
	  void rftfsub(int n, float* a, int offa, int nc, float* c, int startc);
	 
	  void scale_fun(float m, float* a, int offa, bool complex);
	  void cftf040(float* a, int offa);
	  void rftbsub(int n, float* a, int offa, int nc, float* c, int startc);


public:


	int n;

	int *ip;

	float *w;

	int nw;

	int nc;

	//float *wtable;

	//float *wtable_r;

	//float *bk1;

	//float *bk2;

	//Plans plan;

	//int factors[4];

	//float PI;

	//float TWO_PI;

	// local storage which is predeclared
	//float *ak;
	//float *ch;
	//float *ch2;
	//int *nac;
};

