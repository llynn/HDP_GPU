#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "MTRand.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand.h"

#include "npdf.h"
#include "mvpdf_kernel.cu"

#define DELHOST(X) if ((X)!=NULL) {delete [] (X);(X) = NULL;}

NPDF::NPDF() {
    TRUNKSIZE = 65536;

    hX = NULL;
    hMeanAndSigma = NULL;
    hRandomNumber = NULL;
    hComponent = NULL;
    hDensity = NULL;

    //device variables
    dX = NULL;
    dMeanAndSigma = NULL;
    dRandomNumber = NULL;
    dComponent = NULL;
    dDensity = NULL;    
}
NPDF::~NPDF(void)
{
    clear();
}
void NPDF::clear() {
    DELHOST(hX);
    DELHOST(hMeanAndSigma);
    DELHOST(hRandomNumber);
    DELHOST(hComponent);
    DELHOST(hDensity);

    if (dX!=NULL) {
        cudaFree(dX);dX=NULL;
        cudaFree(dMeanAndSigma); dMeanAndSigma = NULL;
        cudaFree(dRandomNumber); dRandomNumber = NULL;
        cudaFree(dComponent); dComponent = NULL;
        cudaFree(dDensity); dDensity = NULL;
    }
}
bool NPDF::SetDevice(int device) {
    //get device count
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { return false; }
    if (device >= deviceCount) {return false;}
    cudaSetDevice(device);
    return true;
}
void NPDF::getPaddedDim() {
    //figure out padding dims
	int MEAN_CHD_DIM = D * (D + 3) / 2	+ 2;
    PACKpad = iAlignUp(MEAN_CHD_DIM,16);
	Dpad = iAlignUp(D,HALFWARP);
    Npad = iAlignUp(N,HALFWARP);
    Tpad = iAlignUp(T,HALFWARP);
}

bool NPDF::AllocateHostMemory() {
    //alocate host memories and prepare host data
    hX = (float*) calloc(Dpad * Npad,sizeof(float));
    hMeanAndSigma = (float*) calloc(PACKpad * Tpad, sizeof(float));
    hRandomNumber = (float*) calloc(Npad,sizeof(float));
    hComponent = (int*) calloc(Npad, sizeof(int));
    hDensity = (float*) calloc(Npad * Tpad , sizeof(float));
    return true;
}
bool NPDF::GetHostData(double *iX) {
    //copy data into host memory
    int i, j;
    if (hX != NULL) {
        float* hDatumPtr = hX;
        for(i=0; i<N; i++) {        //column major
            for(j=0; j<D; j++) {
                hDatumPtr[j] = (float) iX[j*N+i];
            }
            hDatumPtr += Dpad;
        }
    }
    return true;
}

bool NPDF::GetHostData(double *Pi, double *tMu, double *sigma) {
    //copy data into host memory
    int i, j,k;
    int D2 = D*D;
    double temp;
    
    //caculate inverse of Lower chol matrix
    double *iSigma = (double*)calloc(D*D*T,sizeof(double));
    double *logdet = (double*)calloc(T,sizeof(double));
    memcpy(iSigma,sigma,D*D*T * sizeof(double));
    double *Linv = (double*)calloc(D*D*T,sizeof(double));
    for (i = 0; i < T; i++) {   
        L_inv(&Linv[i*D2],&temp,&iSigma[i*D2],D);
        logdet[i] = temp;
    }
    free(iSigma);
    
    int MEAN_CHD_DIM = D * (D + 3) / 2	+ 2;

    //copy densities into host memory
    float* hdptr = hMeanAndSigma;
	for(i=0; i<T; i++) {
		for(j=0; j<D; j++) {
			*hdptr++ =  (float)tMu[j*T+i];
        }
        
		for(k=0; k<D; k++) {       
            for (j = 0; j <=k; j++) {
                *hdptr++ = (float)Linv[i*D2 + j*D + k];
            }
        }

		*hdptr++ = (float)Pi[i];
		*hdptr++ = (float)logdet[i];
		hdptr += ((PACKpad) - (MEAN_CHD_DIM));
	}
    free(Linv);
    free(logdet);
    return true;
}
bool NPDF::UpdateWeight(double *Pi) {
    //copy weight into host memory
    int i;
    int Pos_Pi = D * (D + 3) / 2;
    
    float* hdptr = hMeanAndSigma+Pos_Pi;
	for(i=0; i<T; i++) {
		*hdptr = (float)Pi[i];
		hdptr += PACKpad;
	}
    return true;
}
void NPDF::GetRandNumber(double *rn) {
    float* hranfptr = hRandomNumber;
    for(int i=0; i<N; i++) {        //column major
        hranfptr[i] = (float)rn[i];
    }
    cudaMemcpy(dRandomNumber,hRandomNumber,N * sizeof(float), cudaMemcpyHostToDevice);
}
void NPDF::GetRandNumber(curandGenerator_t &rg) {
    curandGenerateUniform(rg,dRandomNumber,N);
    cudaThreadSynchronize();
}

void NPDF::CopyToGPU(int alldata) {
    //copy data from host to device
    if (alldata > 0) {
        cudaMemcpy(dX,hX,N * Dpad * sizeof(float),cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dMeanAndSigma,hMeanAndSigma,T * PACKpad * sizeof(float), cudaMemcpyHostToDevice);	
}
bool NPDF::DoZ_max_no_sampling(double *Z, double *WWk,NPDF *theother, int transpose) {
    
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    
    if (transpose > 0) {
        for (int i =0; i < T1; i++) {
            for (int j = 0; j < T; j++) {
                WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
            }
        }
    } else {
        for (int i =0; i < T1; i++) {   //no transpose here, just padding
            for (int j = 0; j < T; j++) {
                WWkPad[i*Tpad+j] = (float)WWk[i*T+j];
            }
        }
    }
    
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }

    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    
    //prepare indicator vector
    unsigned int  *ZPad = new unsigned int[Npad];
    for (int i =0; i < N; i++) {
        ZPad[i] = (unsigned int)Z[i]-1; //going back to 0-based
    }

    unsigned int *dZ = NULL;
    if ( cudaMalloc( (void**) &dZ, Npad * sizeof(unsigned int)) != cudaSuccess )  {
        return false;
    }

    cudaMemcpy(dZ,ZPad,N * sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(dDensity,hDensity,N * Tpad* sizeof(float),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    
    dim3 threads(HALFWARP, HALFWARP); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    Matrix_dot_row_inplace_byC<<<grid, threads>>>(dDensity, dZ, dWWk, N, Tpad, T);
    
    DoMax();
    
    delete [] ZPad;
    delete [] WWkPad;
    cudaFree(dWWk);
    cudaFree(dZ);
    return true;
}

bool NPDF::DoZ(double *Z, double *WWk,NPDF *theother, int transpose) {
    
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    
    if (transpose > 0) {
        for (int i =0; i < T1; i++) {
            for (int j = 0; j < T; j++) {
                WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
            }
        }
    } else {
        for (int i =0; i < T1; i++) {   //no transpose here, just padding
            for (int j = 0; j < T; j++) {
                WWkPad[i*Tpad+j] = (float)WWk[i*T+j];
            }
        }
    }
    
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }

    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    
    //prepare indicator vector
    unsigned int  *ZPad = new unsigned int[Npad];
    for (int i =0; i < N; i++) {
        ZPad[i] = (unsigned int)Z[i]-1; //going back to 0-based
    }

    unsigned int *dZ = NULL;
    if ( cudaMalloc( (void**) &dZ, Npad * sizeof(unsigned int)) != cudaSuccess )  {
        return false;
    }

    cudaMemcpy(dZ,ZPad,N * sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(dDensity,hDensity,N * Tpad* sizeof(float),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    
    dim3 threads(HALFWARP, HALFWARP); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    Matrix_dot_row_inplace_byC<<<grid, threads>>>(dDensity, dZ, dWWk, N, Tpad, T);
    
    NormalizeDensity();   
    DoSample(0);  
 
    delete [] ZPad;
    delete [] WWkPad;
    cudaFree(dWWk);
    cudaFree(dZ);
    return true;
}

bool NPDF::DoPDFandSample_C(double *Z, double *WWk,NPDF *theother) {
    
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    for (int i =0; i < T1; i++) {
        for (int j = 0; j < T; j++) {
            WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
        }
    }
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    
    //prepare indicator vector
    unsigned int  *ZPad = new unsigned int[Npad];
    for (int i =0; i < N; i++) {
        ZPad[i] = (unsigned int)Z[i]-1; //going back to 0-based
    }
    unsigned int *dZ = NULL;
    if ( cudaMalloc( (void**) &dZ, Npad * sizeof(unsigned int)) != cudaSuccess )  {
        return false;
    }
    cudaMemcpy(dZ,ZPad,Npad * sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaThreadSynchronize(); 
    DoPDF(0,0);

    dim3 threads(HALFWARP, HALFWARP); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    Matrix_dot_row_inplace_byC<<<grid, threads>>>(dDensity, dZ, dWWk, N, Tpad, T);
    
    //cudaMemcpy(hDensity, dDensity, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);

    NormalizeDensity();
    DoSample(0);  
   
    delete [] ZPad;
    delete [] WWkPad;
    cudaFree(dWWk);
    cudaFree(dZ);
    return true;
}
void NPDF::DoPDF(int islogscale, int backtohost) {
    //call gpu kermnels
    dim3 gridPDF(N/DATA_IN_BLOCK, T/DENSITIES_IN_BLOCK);
    if (N % DATA_IN_BLOCK != 0)
        gridPDF.x += 1;
    if (T % DENSITIES_IN_BLOCK != 0)
        gridPDF.y += 1;
    dim3 blockPDF(DATA_IN_BLOCK,DENSITIES_IN_BLOCK);
    int sharedMemSize = (DENSITIES_IN_BLOCK * PACKpad + DATA_IN_BLOCK * D + DENSITIES_IN_BLOCK*DATA_IN_BLOCK) * sizeof(float);
    mvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(dX,dMeanAndSigma,dDensity,D, N, T,islogscale);
    cudaThreadSynchronize(); 
    if (backtohost>0) {
        cudaMemcpy(hDensity, dDensity, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);
    }
}
void NPDF::DoMax() { //will destroy the density matrix
    dim3 gridMax(N/SAMPLE_BLOCK,1);
    if (N % SAMPLE_BLOCK != 0)
        gridMax.x +=1 ;
    dim3 blockMax(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
    GetMaxFromMeasureMedium<<<gridMax, blockMax>>>(dDensity,dComponent,N,T);
    cudaThreadSynchronize(); 
    //copy result back to host
    cudaMemcpy(hComponent, dComponent, N* sizeof(int), cudaMemcpyDeviceToHost);
}

void NPDF::DoSample(int islogscale) { //will destroy the density matrix
    dim3 gridSample(N/SAMPLE_BLOCK,1);
    if (N % SAMPLE_BLOCK != 0)
        gridSample.x +=1 ;
    dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
    sampleFromMeasureMedium<<<gridSample, blockSample>>>(dDensity,dRandomNumber,
            dComponent,N,T,islogscale);
    cudaThreadSynchronize(); 
    //copy result back to host
    cudaMemcpy(hComponent, dComponent, N* sizeof(int), cudaMemcpyDeviceToHost);
}

void NPDF::NormalizeDensity() {
    dim3 gridSample(N/SAMPLE_BLOCK,1);
    if (N % SAMPLE_BLOCK != 0)
        gridSample.x +=1 ;
    dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
    Normalize<<<gridSample, blockSample>>>(dDensity,N,T);
    cudaThreadSynchronize(); 
    cudaMemcpy(hDensity, dDensity, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);
}

void NPDF::NormalizeDensity_log() { //temp function
    dim3 gridSample(N/SAMPLE_BLOCK,1);
    if (N % SAMPLE_BLOCK != 0)
        gridSample.x +=1 ;
    dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
    Normalize_log<<<gridSample, blockSample>>>(dDensity,N,T);
    cudaThreadSynchronize(); 
    cudaMemcpy(hDensity, dDensity, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);
}


float NPDF::SumLog(double *Z4) {
    float s  = 0.0;
    for (int i = 0; i < N; i++) {
        s += log(hDensity[i*Tpad + (unsigned int)Z4[i]-1]);
    }
    return s;
}
float NPDF::Sum(double *Z4) {
    float s  = 0.0;
    for (int i = 0; i < N; i++) {
        s += hDensity[i*Tpad + (unsigned int)Z4[i]-1];
    }
    return s;
}

//this one is similar to that of DoZ1Z2
bool NPDF::DoAltQQ(double *WWk, NPDF *theother, double *Z2) {
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    for (int i =0; i < T1; i++) {
        for (int j = 0; j < T; j++) {
            WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
        }
    }
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);
    
    theother->DoPDF(0,0); 
    theother->NormalizeDensity();
    
    float *dProb = NULL;    
    if ( cudaMalloc( (void**) &dProb, Npad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    dim3 threads(BLOCKSIZE, BLOCKSIZE); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    matrixMul<BLOCKSIZE><<<grid,threads>>>(dProb,theother->dDensity,dWWk, T1pad,Tpad, N, T1, T);
    cudaThreadSynchronize();
    
    float *hProb = (float*) calloc(Npad * Tpad , sizeof(float));
    cudaMemcpy(hProb, dProb, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);

    QQ = 0.0;
    for (int i = 0; i < N; i++) {
        QQ += log(hProb[i*Tpad +(int)Z2[i]-1]);//going back to zero-based
    }
    cudaFree(dProb); dProb = NULL;
    
    delete [] hProb;
    delete [] WWkPad;
    cudaFree(dWWk);
    return true;

}

bool NPDF::DoZ5(double *WWk, NPDF *theother, int what) {
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    for (int i =0; i < T1; i++) {
        for (int j = 0; j < T; j++) {
            WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
        }
    }
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);

    theother->DoPDF(0,0); 
    theother->NormalizeDensity();
    
    float *dProb = NULL;    
    if ( cudaMalloc( (void**) &dProb, Npad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    dim3 threads(BLOCKSIZE, BLOCKSIZE); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    matrixMul<BLOCKSIZE><<<grid,threads>>>(dProb,theother->dDensity,dWWk, T1pad,Tpad, N, T1, T);
    cudaThreadSynchronize();
    
    if (what <=0) {
        DoPDF(0,0); 
        dim3 grid1(Npad/HALFWARP,Tpad/HALFWARP);
        dim3 thread1(HALFWARP,HALFWARP);
        matdot_ip<<<grid1,thread1>>>(dDensity,dProb,N,T,Tpad);
        NormalizeDensity();
        //this destroies the density matrix on device
        if (what == -1) {
            DoMax();
        } else { //what = zero
            DoSample(0);  
        }
        cudaMemcpy(hDensity, dProb, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);   //output dProb instead
    } else {
        cudaMemcpy(dDensity, dProb, N * Tpad* sizeof(float), cudaMemcpyDeviceToDevice);   //output dProb instead
        NormalizeDensity_log();
    }
  
    cudaFree(dProb); dProb = NULL;
    
    delete [] WWkPad;
    cudaFree(dWWk);

    return true;

}

bool NPDF::DoZ1Z2(double *WWk, NPDF *theother) {
    //prepare transition matrix
    int T1 = theother->T;
    int T1pad = theother->Tpad;          
    float *WWkPad = new float[T1pad * Tpad];
    memset(WWkPad,0,T1pad * Tpad*sizeof(float));
    for (int i =0; i < T1; i++) {
        for (int j = 0; j < T; j++) {
            WWkPad[i*Tpad+j] = (float)WWk[j*T1+i];
        }
    }
    float *dWWk = NULL;
    if ( cudaMalloc( (void**) &dWWk, T1pad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    cudaMemcpy(dWWk,WWkPad,T1pad * Tpad * sizeof(float),cudaMemcpyHostToDevice);

    theother->DoPDF(0,0); 
    theother->NormalizeDensity();
    
    float *dProb = NULL;    
    if ( cudaMalloc( (void**) &dProb, Npad * Tpad * sizeof(float)) != cudaSuccess )  {
        return false;
    }
    dim3 threads(BLOCKSIZE, BLOCKSIZE); 
    dim3 grid(Tpad / threads.x, Npad / threads.y);
    matrixMul<BLOCKSIZE><<<grid,threads>>>(dProb,theother->dDensity,dWWk, T1pad,Tpad, N, T1, T);
    cudaThreadSynchronize();
    
    
    DoPDF(0,0); 
    
    dim3 grid1(Npad/HALFWARP,Tpad/HALFWARP);
    dim3 thread1(HALFWARP,HALFWARP);
    matdot_ip<<<grid1,thread1>>>(dDensity,dProb,N,T,Tpad);
    NormalizeDensity();

    float *hProb = (float*) calloc(Npad * Tpad , sizeof(float));
    cudaMemcpy(hProb, dProb, N * Tpad* sizeof(float), cudaMemcpyDeviceToHost);

    //this destroies the density matrix on device
    DoSample(0);  
    theother->DoSample(0);
    
    QQ = 0.0;

    for (int i = 0; i < N; i++) {
        QQ += log(hProb[i*Tpad +hComponent[i]]);
    }

    cudaFree(dProb); dProb = NULL;
    
    delete [] hProb;
    delete [] WWkPad;
    cudaFree(dWWk);

    return true;
}
int NPDF::iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}
void NPDF::L_inv(double y[], double *logdet, double x[], mwSize m)
{
   int row,col,row1;
   double temp;
   int i,j,k;
   
   double *xx = (double*) calloc(m*m,sizeof(double));
   memset(xx,0,m*m*sizeof(double));
   *logdet = 0.0;
   for (i=0; i<m;i++) {
        for (j = 0; j < i;j++) {
            temp = x[i+j*m];
            for (k=0; k <j;k++) {
                temp-=xx[i+k*m] * xx[j+k*m];
            }
            xx[i+j*m] = temp / xx[j+j*m];
        }
        temp = x[i+i*m];
        for (k=0; k <i;k++) {
            temp-=xx[i+k*m] * xx[i+k*m];
        }
        xx[i+i*m] = sqrt(temp);
        *logdet += log(xx[i+i*m]);
   }
   *logdet *= 2.0;
	
   //invert a lower triangular matrix
   //memcpy(xx, x, sizeof(double) * m * m);
   for (row = 0; row < m; row++) {
       int index = row * m + row;
       double temp = 1.0/xx[index];
       y[index] = temp;
       xx[index] = 1.0;
       for (col = 0; col < row; col ++) {
           xx[col * m + row] *= temp;
       }
   } 
   for (row = 0; row < m; row++) {
       for(col = 0; col <=row; col++) {
           int nbase = col*m;
            for (row1 = row+1;row1 < m; row1++) {
                temp = xx[row1+row*m]/ xx[row+row*m];
                xx[row1+nbase] -=xx[row+nbase] * temp;
                y[row1+nbase] -=y[row+nbase] * temp;
            }
        }
   }
   free(xx);
}

bool NPDF::InitGPU() {
    if (dX==NULL) {
        if ( cudaMalloc( (void**) &dX, Dpad * Npad*sizeof(float) ) != cudaSuccess) {
            return false;
        }
        if ( cudaMalloc( (void**) &dMeanAndSigma, PACKpad * Tpad * sizeof(float)) != cudaSuccess )  {
            return false;
        }
        if ( cudaMalloc( (void**) &dRandomNumber, Npad * sizeof(float)) != cudaSuccess ) {
            return false;
        } 
        if ( cudaMalloc( (void**) &dComponent, Npad * sizeof(int)) != cudaSuccess )  {
            return false;
        }
        if ( cudaMalloc( (void**) &dDensity, Npad * Tpad * sizeof(float)) != cudaSuccess )  {
            return false;
        }
    }
    return true;
}
void NPDF::getDensity(float * r) {
    for (int i = 0; i <N; i++) {        //column major
        for (int j = 0; j < T; j++) {
            r[j*N+i] = hDensity[i*Tpad+j];
        }
    }
}
void NPDF::getIndicator(float * r) {
    for (int i = 0; i <N; i++) {
        r[i] = hComponent[i]+1;
    }
}


