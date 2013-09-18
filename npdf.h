/* npdf.h
 * @author Quanli Wang, quanli@stat.duke.edu
 */
#pragma once
class NPDF
{
public:
	NPDF();
    ~NPDF(void);
public:
    
    int N,D,T;
    int Dpad,PACKpad,Npad,Tpad;
    
    int TRUNKSIZE;
	//host variables
    float* hX;
    float* hMeanAndSigma;
    float* hRandomNumber;
    int* hComponent;
    float* hDensity;
    
    double QQ;

    //device variables
    float* dX;
    float* dMeanAndSigma;
    float* dRandomNumber;
    int* dComponent;
    float* dDensity;
    
    //working variable
    void clear();
    void getPaddedDim();
    bool InitGPU();
    bool SetDevice(int device);
    bool AllocateHostMemory();
    bool GetHostData(double *x);
    bool GetHostData(double *pi, double *mu, double *sigma);
    bool UpdateWeight(double *pi);
    void CopyToGPU(int alldata);
    void DoPDF(int islogscale, int backtohost);
    bool DoPDFandSample_C(double *Z, double *WWk,NPDF *theother);
    bool DoZ(double *Z, double *WWk,NPDF *theother, int transpose);
    bool DoZ_max_no_sampling(double *Z, double *WWk,NPDF *theother, int transpose);
    void DoSample(int islogscale);
    void DoMax();
    void NormalizeDensity();
    void NormalizeDensity_log();
    float SumLog(double *Z4);
    float Sum(double *Z4);
    
    void GetRandNumber(double *rn);
    void GetRandNumber(curandGenerator_t &rg);
    bool DoZ1Z2(double *WWk, NPDF *theother);
    bool DoAltQQ(double *WWk, NPDF *theother, double *Z2);
    bool DoZ5(double *WWk, NPDF *theother, int what);
    
    void L_inv(double y[], double *logdet, double x[], mwSize m);
    void getDensity(float * r);
    void getIndicator(float * r);
    int iAlignUp(int a, int b);
};
