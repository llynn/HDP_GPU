
#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "npdf.cu"

#include <vector>
#include <map>
#include <set>

using namespace std;

#define MAXX 10
NPDF X[MAXX];

MTRand mt;
curandGenerator_t gen;
int initGPI = 0;

void GetString(const mxArray *string_array_ptr, char **buf) {
    size_t buflen;
    if ( mxIsChar(string_array_ptr) != 1)
        mexErrMsgTxt( "First input must be a string");
    if (mxGetM(string_array_ptr)!=1)
        mexErrMsgTxt( "First input must be a row vector");
    buflen = (mxGetM(string_array_ptr) * mxGetN(string_array_ptr)) + 1;
    *buf=(char*)mxCalloc(buflen, sizeof(char));
    *buf = mxArrayToString(string_array_ptr);
    //mexPrintf("%s\n",*buf);

}
//Input Check
int AddData(int nlhs,  mxArray *plhs[],int nrhs, const mxArray *prhs[], NPDF* Data) {
    const mwSize *dimsX,*dimsPi,*dimsSigma;
    mwSize  ndim; 
    if(nrhs!=6) {
        mexErrMsgTxt("action, data_set_index, X,Pi,mu,Sigma are needed.");
    }
    int dataset = (int)mxGetScalar(prhs[1]);
    if (dataset >= MAXX) {
        mexErrMsgTxt("data_set_index is too big.");
    }
    
    //check dimmension of x
    ndim = mxGetNumberOfDimensions(prhs[2]);
    if (ndim != 2) {
        mexErrMsgTxt("input data must be a 2D array");
    }    
    dimsX = mxGetDimensions(prhs[2]);
    Data[dataset].N = dimsX[0];
    Data[dataset].D = dimsX[1];
    
    //check dimension of pi
    ndim = mxGetNumberOfDimensions(prhs[3]);
    dimsPi = mxGetDimensions(prhs[3]);
    if (ndim==2 && (dimsPi[0]>1) && (dimsPi[1] > 1)) {
        mexErrMsgTxt("Pi must be a vector\n");
    }
    Data[dataset].T = mxGetNumberOfElements(prhs[3]); 
    
    //check mu
    ndim = mxGetNumberOfDimensions(prhs[4]);
    if (ndim != 2) {
        mexErrMsgTxt("mu must be a 2D array");
    }
    if (mxGetM(prhs[4]) != Data[dataset].T || mxGetN(prhs[4]) != Data[dataset].D) {
        mexErrMsgTxt("mu dimmensions do not match");
    }

    //check sigma
    ndim = mxGetNumberOfDimensions(prhs[5]);
    dimsSigma = mxGetDimensions(prhs[5]);
    if (ndim != 3) {
        mexErrMsgTxt("Sigma must be a 3D array");
    }
    if (dimsSigma[0] != Data[dataset].D || dimsSigma[1] !=Data[dataset].D || dimsSigma[2] != Data[dataset].T) {
        mexErrMsgTxt("Sigma dimmensions do not match");
    }
    return dataset;
}
int UpdateCluster(int nrhs, const mxArray *prhs[], NPDF* Data) {
    const mwSize *dimsPi,*dimsSigma;
    mwSize  ndim; 
    if(nrhs!=5) {
        mexErrMsgTxt("action, data_set_index, Pi,mu,Sigma are needed.");
    }
    int dataset = (int)mxGetScalar(prhs[1]);
    if (dataset >= MAXX) {
        mexErrMsgTxt("data_set_index is too big.");
    }

    //check dimension of pi
    ndim = mxGetNumberOfDimensions(prhs[2]);
    dimsPi = mxGetDimensions(prhs[2]);
    if (ndim==2 && (dimsPi[0]>1) && (dimsPi[1] > 1)) {
        mexErrMsgTxt("Pi must be a vector\n");
    }
    if (Data[dataset].T != mxGetNumberOfElements(prhs[2])) {
        mexErrMsgTxt("Pi does not match\n");
    }
    
    //check mu
    ndim = mxGetNumberOfDimensions(prhs[3]);
    if (ndim != 2) {
        mexErrMsgTxt("mu must be a 2D array");
    }
    if (mxGetM(prhs[3]) != Data[dataset].T || mxGetN(prhs[3]) != Data[dataset].D) {
        mexErrMsgTxt("mu dimmensions do not match");
    }

    //check sigma
    ndim = mxGetNumberOfDimensions(prhs[4]);
    dimsSigma = mxGetDimensions(prhs[4]);
    if (ndim != 3) {
        mexErrMsgTxt("Sigma must be a 3D array");
    }
    if (dimsSigma[0] != Data[dataset].D || dimsSigma[1] !=Data[dataset].D || dimsSigma[2] != Data[dataset].T) {
        mexErrMsgTxt("Sigma dimmensions do not match");
    }
    return dataset;
}


// ----------------- the MEX driver runs on the CPU --------------------
void mexFunction(int nlhs,  mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    if( nrhs < 1 ) {
		mexErrMsgTxt( "Usage: at least one parameter is needed");
    }
    char *cmd = NULL;
    //char ** dataitems = NULL;
    GetString(prhs[0],&cmd);
    string strcmd(cmd);
    if (strcmd == "setdevice") {
        if (initGPI ==0) {
            if (nrhs!=3) { 
                mexErrMsgTxt( "Device and random seed are needed");
            }
            int device = (int)mxGetScalar(prhs[1]);
            int randseed = (int)mxGetScalar(prhs[2]);
            if (!X[0].SetDevice(device)) {
                mexErrMsgTxt( "Failed to set GPU device.");
            }
            mt.seed(randseed); 
            if (curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT) != cudaSuccess )  {
                mexErrMsgTxt( "Failed to initilize GPU random number generators.");
            }
            if (curandSetPseudoRandomGeneratorSeed(gen, randseed) != cudaSuccess) {
                mexErrMsgTxt( "Failed to initilize GPU random number generators.");
            }
            initGPI = 1;
        } else {
            mexErrMsgTxt( "Device can only be set once");
        }
    } else if (strcmd == "clear") {
        if (initGPI > 0) {
            for (int i =0; i < MAXX; i++) {
                X[i].clear();
            }
            curandDestroyGenerator(gen);
        }
        initGPI = 0;
        
    } else if (strcmd == "adddata") {
        if (initGPI >0) {
            int dataset = AddData(nlhs,plhs,nrhs,prhs, X);    
            X[dataset].clear();
            X[dataset].getPaddedDim();
            X[dataset].AllocateHostMemory();

            double *iX = mxGetPr(prhs[2]);
            double *Pi = mxGetPr(prhs[3]);
            double *tMu = mxGetPr(prhs[4]);
            double *sigma = mxGetPr(prhs[5]);
            X[dataset].GetHostData(iX);
            X[dataset].GetHostData(Pi, tMu,sigma);
            X[dataset].InitGPU();
            X[dataset].CopyToGPU(1);
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "updatecluster") {
        if (initGPI >0) {
            int dataset = UpdateCluster(nrhs,prhs, X);    
            double *Pi = mxGetPr(prhs[2]);
            double *tMu = mxGetPr(prhs[3]);
            double *sigma = mxGetPr(prhs[4]);
            X[dataset].GetHostData(Pi, tMu,sigma);
            X[dataset].CopyToGPU(0);
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "pdf") {
        if (initGPI >0) {
            if (nrhs!=3) { 
                mexErrMsgTxt( "Usage:npdf('pdf',dataset,logscale)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            int logscale = (int)mxGetScalar(prhs[2]);
            X[dataset].DoPDF(logscale,nlhs); 
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset].N, X[dataset].T, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset].getDensity(r);
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "sum") {
        if (initGPI >0) {
            if (nrhs!=3) { 
                mexErrMsgTxt( "Usage:npdf('sum',dataset,Z)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            double *Z = mxGetPr(prhs[2]);
            float result = X[dataset].Sum(Z);
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
                double* r = (double*)mxGetData(plhs[0]);
                r[0] = result;
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "pdf&sum") {
        if (initGPI >0) {
            if (nrhs!=3) { 
                mexErrMsgTxt( "Usage:npdf('pdf',dataset,z4)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            double *Z4 = mxGetPr(prhs[2]);
            X[dataset].DoPDF(0,0); 
            X[dataset].NormalizeDensity();
            float result = X[dataset].SumLog(Z4);
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
                double* r = (double*)mxGetData(plhs[0]);
                r[0] = result;
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "density") {       //read computed density only
        if (initGPI >0) {
            if (nrhs!=2) { 
                mexErrMsgTxt( "Usage:gnpdf('density',dataset)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset].N, X[dataset].T, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset].getDensity(r);
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "sample") {
        if (initGPI >0) {
            if (nrhs<3) { 
                mexErrMsgTxt( "Usage:npdf('sample',dataset,logscale,(optional)randomnumer)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            int logscale = (int)mxGetScalar(prhs[2]);
            if (nrhs > 3) { //random number providedes
                //check random numbers
                mwSize ndim = mxGetNumberOfDimensions(prhs[3]);
                if (ndim==2 && (mxGetM(prhs[3])>1) && (mxGetN(prhs[3]) > 1)) {
                    mexErrMsgTxt("randomnumber must be a vector\n");
                }
                int temp = mxGetNumberOfElements(prhs[3]); 
                if (temp != X[dataset].N) {
                    mexErrMsgTxt("randomnumber dimmension does not match\n");
                }
                double *irn = mxGetPr(prhs[3]);
                X[dataset].GetRandNumber(irn);
            } else {
                X[dataset].GetRandNumber(gen);
            }
            X[dataset].DoSample(logscale); 
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset].getIndicator(r);
            } 
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "max") {
        if (initGPI >0) {
            if (nrhs<2) { 
                mexErrMsgTxt( "Usage:npdf('max',dataset");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            X[dataset].DoMax(); 
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset].getIndicator(r);
            } 
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "pdf&sample") {
        if (initGPI >0) {
            if (nrhs<2) { 
                mexErrMsgTxt( "Usage:npdf('pdf&sample',dataset,(optional)randomnumber)");
            }
            int dataset = (int)mxGetScalar(prhs[1]);
            if (nrhs > 2) { //random number providedes
                //check random numbers
                mwSize ndim = mxGetNumberOfDimensions(prhs[2]);
                if (ndim==2 && (mxGetM(prhs[2])>1) && (mxGetN(prhs[2]) > 1)) {
                    mexErrMsgTxt("randomnumber must be a vector\n");
                }
                int temp = mxGetNumberOfElements(prhs[2]); 
                if (temp != X[dataset].N) {
                    mexErrMsgTxt("randomnumber dimmension does not match\n");
                }
                double *irn = mxGetPr(prhs[2]);
                X[dataset].GetRandNumber(irn);
            } else {
                X[dataset].GetRandNumber(gen);
            }
            X[dataset].DoPDF(1,0); //logscale, no backtohost

            X[dataset].DoSample(1); 
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset].getIndicator(r);
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    }  else if (strcmd == "pdf&sample-c") {
        if (initGPI >0) {
            if (nrhs<5) { 
                mexErrMsgTxt( "Usage:npdf('pdf&sample-c',dataset1,dataset2,Z,W,(optional)randomnumber)");
            }
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            double *Z1 = mxGetPr(prhs[3]);
            double *wwk = mxGetPr(prhs[4]); 
            if (nrhs > 5) { //random number provideded
                double *irn = mxGetPr(prhs[5]);
                X[dataset2].GetRandNumber(irn);
            } else {
                X[dataset2].GetRandNumber(gen);
            }
            X[dataset2].DoPDFandSample_C(Z1,wwk,&X[dataset1]);
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset2].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset2].getIndicator(r);
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "zs") {
        if (initGPI >0) {
            if (nrhs<5) { 
                mexErrMsgTxt( "Usage:npdf('zs',dataset1,dataset2,W,Z4");
            }
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            double *wwk = mxGetPr(prhs[3]); 
            double *Z4 = mxGetPr(prhs[4]);

            //update Z1
            X[dataset1].GetRandNumber(gen);
            X[dataset1].DoPDF(0,0); //logscale, no backtohost
            X[dataset1].NormalizeDensity();
            X[dataset1].DoSample(0); 
            plhs[0]=mxCreateNumericMatrix(X[dataset1].N, 1, mxSINGLE_CLASS, mxREAL);
            float* r = (float*)mxGetData(plhs[0]);
            X[dataset1].getIndicator(r);
            
            
            //update Z2 using old Z4
            X[dataset2].GetRandNumber(gen);
            X[dataset2].DoPDF(0,0); 
            X[dataset2].NormalizeDensity();
            X[dataset2].DoZ(Z4,wwk,&X[dataset1], 1);
            plhs[1]=mxCreateNumericMatrix(X[dataset2].N, 1, mxSINGLE_CLASS, mxREAL);
            r = (float*)mxGetData(plhs[1]);
            X[dataset2].getIndicator(r);

            //update Z4 using new Z2
            X[dataset1].GetRandNumber(gen);
            for (int i = 0; i <X[dataset1].N; i++) {
                Z4[i] = r[i];
            }
            X[dataset1].DoZ(Z4,wwk,&X[dataset2], 0);
            plhs[2]=mxCreateNumericMatrix(X[dataset1].N, 1, mxSINGLE_CLASS, mxREAL);
            r = (float*)mxGetData(plhs[2]);
            X[dataset1].getIndicator(r);
            

        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "zz") {
        if (initGPI >0) {
            if (nrhs<6) { 
                mexErrMsgTxt( "Usage:npdf('zz',dataset1,dataset2,W,Z5,what");
            }
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            double *wwk = mxGetPr(prhs[3]); 
            double *Z5 = mxGetPr(prhs[4]);
            int what = (int)mxGetScalar(prhs[5]);

            //update Z1
            X[dataset1].GetRandNumber(gen);
            X[dataset1].DoPDF(0,0); //logscale, no backtohost
            X[dataset1].NormalizeDensity();
            if (what ==-1) {
                X[dataset1].DoMax(); 
            } else {
                X[dataset1].DoSample(0); 
            }
            plhs[0]=mxCreateNumericMatrix(X[dataset1].N, 1, mxSINGLE_CLASS, mxREAL);
            float* r = (float*)mxGetData(plhs[0]);
            X[dataset1].getIndicator(r);
            
            //update Z2 using Z5
            X[dataset1].DoZ_max_no_sampling(Z5,wwk,&X[dataset2], 0);
            plhs[1]=mxCreateNumericMatrix(X[dataset1].N, 1, mxSINGLE_CLASS, mxREAL);
            r = (float*)mxGetData(plhs[1]);
            X[dataset1].getIndicator(r);
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "z5") {
        if (initGPI >0) {
            if (nrhs<5) { 
                mexErrMsgTxt( "Usage:npdf('z5',dataset1,dataset2,W,action");
            }
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            double *wwk = mxGetPr(prhs[3]); 
            int what = (int)mxGetScalar(prhs[4]);
            X[dataset2].GetRandNumber(gen);
            X[dataset2].DoZ5(wwk,&X[dataset1],what);
            if (what <=0) { //-1 for max, 0 for density
                plhs[0]=mxCreateNumericMatrix(X[dataset2].N, 1, mxSINGLE_CLASS, mxREAL);
                float *r = (float*)mxGetData(plhs[0]);
                X[dataset2].getIndicator(r);                
            } else {
                if (nlhs > 0) {
                    plhs[0]=mxCreateNumericMatrix(X[dataset2].N, X[dataset2].T, mxSINGLE_CLASS, mxREAL);
                    float* r = (float*)mxGetData(plhs[0]);
                    X[dataset2].getDensity(r);
                }
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } else if (strcmd == "qq") {
        if (initGPI >0) {
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            if (nrhs ==6) { //need to reavluate the pdf
                //check indicators 
                int temp = mxGetNumberOfElements(prhs[3]); 
                if (temp != X[dataset1].T) {
                    mexErrMsgTxt("indicator dimmension does not match\n");
                }
                double *weight = mxGetPr(prhs[3]);
                X[dataset1].UpdateWeight(weight);
                X[dataset1].CopyToGPU(0);
                double *wwk = mxGetPr(prhs[4]); 
                double *Z2 = mxGetPr(prhs[5]); 
                X[dataset2].DoAltQQ(wwk, &X[dataset1],Z2);
                
            } else {
                mexErrMsgTxt( "Usage:npdf('qq',dataset1,dataset2,Weight_alt,WWK0,Z2)");
            }
            
            if (nlhs==1) {
                plhs[0]=mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
                double* r = mxGetPr(plhs[0]);
                r[0] = X[dataset2].QQ;
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    }else if (strcmd == "z1z2") {
        if (initGPI >0) {
            if (nrhs<4) { 
                mexErrMsgTxt( "Usage:npdf('z1z2',dataset1,dataset2,wwk0,(optional)randomnumber1,(optional)randomnumber2)");
            }
            int dataset1 = (int)mxGetScalar(prhs[1]);
            int dataset2 = (int)mxGetScalar(prhs[2]);
            if (nrhs > 4) { //random number providedes
                //check random numbers1
                mwSize ndim = mxGetNumberOfDimensions(prhs[4]);
                if (ndim==2 && (mxGetM(prhs[4])>1) && (mxGetN(prhs[4]) > 1)) {
                    mexErrMsgTxt("randomnumber1 must be a vector\n");
                }
                int temp = mxGetNumberOfElements(prhs[4]); 
                if (temp != X[dataset1].N) {
                    mexErrMsgTxt("randomnumber1 dimmension does not match\n");
                }
                double *irn = mxGetPr(prhs[4]);
                X[dataset1].GetRandNumber(irn);

                //check random numbers1
                ndim = mxGetNumberOfDimensions(prhs[5]);
                if (ndim==2 && (mxGetM(prhs[5])>1) && (mxGetN(prhs[5]) > 1)) {
                    mexErrMsgTxt("randomnumber2 must be a vector\n");
                }
                temp = mxGetNumberOfElements(prhs[5]); 
                if (temp != X[dataset2].N) {
                    mexErrMsgTxt("randomnumber2 dimmension does not match\n");
                }
                irn = mxGetPr(prhs[5]);
                X[dataset2].GetRandNumber(irn);

            } else {
                X[dataset1].GetRandNumber(gen);
                X[dataset2].GetRandNumber(gen);
            }
            
            double *w = mxGetPr(prhs[3]);  
            if (!X[dataset2].DoZ1Z2(w,&X[dataset1])) {
                 mexErrMsgTxt("Allocating device memory failed\n");
            }
            
            if (nlhs>=1) {
                plhs[0]=mxCreateNumericMatrix(X[dataset1].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[0]);
                X[dataset1].getIndicator(r);
            }
            if (nlhs>=2) {
                plhs[1]=mxCreateNumericMatrix(X[dataset2].N, 1, mxSINGLE_CLASS, mxREAL);
                float* r = (float*)mxGetData(plhs[1]);
                X[dataset2].getIndicator(r);
            }
            if (nlhs>=3) {
                plhs[2]=mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
                double* r = mxGetPr(plhs[2]);
                r[0] = X[dataset2].QQ;
            }
        } else {
            mexErrMsgTxt( "Device has to be set first");
        }
    } 
}