/* gpumvpdf.cu
 * @author Quanli Wang quanli@stat.duke.du, 2010
 */

#include "cuda.h"
#include "cuda_runtime.h"
#define HALFWARP	16
#define DENSITIES_IN_BLOCK		HALFWARP  //has to be 16 if TWISTED_DENSITY is defined
#define	DATA_IN_BLOCK			HALFWARP  //has to be 16 if TWISTED_DENSITY is defined
#define SAMPLE_BLOCK			32	
#define SAMPLE_DENSITY_BLOCK	HALFWARP
#define LOG_2_PI 1.83787706640935
#define REAL float


#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#define TILE_DIM    HALFWARP
#define BLOCK_ROWS  HALFWARP
#define BLOCKSIZE   HALFWARP

//Align a to nearest higher multiple of b
extern "C" __device__ int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//A = A.*B, Padded in rows
__global__ void matdot_ip(float *A, float *B, int N, int P, int Ppad) {
    const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
    const int indexN = blockIdx.x * blockDim.x + thidx;
    const int indexP = blockIdx.y * blockDim.y + thidy;
    if (indexN < N && indexP < P) {
        int c = indexN * Ppad + indexP;
        float a = A[c];
        float b = B[c];
        __syncthreads();
        A[c] = a*b;
    }
    __syncthreads();
}

template <int BLOCK_SIZE> __global__ void
matrixMul(float* C, float* A, float* B, int K, int N, int M0, int K0, int N0)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = K * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + K - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * N;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;
	
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
	
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    if (BLOCK_SIZE * (by+1) <= M0 && BLOCK_SIZE * (bx+1) <= N0) {
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
            AS(ty, tx) = A[a + K * ty + tx];
            BS(ty, tx) = B[b + N * ty + tx];
            __syncthreads();
            if (a-aBegin + BLOCK_SIZE-1 < K0) {
                for (int k = 0; k < BLOCK_SIZE; ++k) {	
                    Csub += AS(ty, k) * BS(k, tx);
                }
            } else {
                for (int k = 0; k < BLOCK_SIZE; ++k) {	
                    if (a-aBegin+k<K0) {
                        Csub += AS(ty, k) * BS(k, tx);
                    }
                }         
            }
            
            __syncthreads();
        }
    } else {
        for (int a = aBegin, b = bBegin;
                 a <= aEnd;
                 a += aStep, b += bStep) {
            AS(ty, tx) = A[a + K * ty + tx];
            BS(ty, tx) = B[b + N * ty + tx];
            __syncthreads();

            if (BLOCK_SIZE * by + ty < M0 && BLOCK_SIZE * bx + tx < N0) {
                if (a-aBegin + BLOCK_SIZE-1 < K0) {
                    for (int k = 0; k < BLOCK_SIZE; ++k) {	
                        Csub += AS(ty, k) * BS(k, tx);
                    }
                } else {
                    for (int k = 0; k < BLOCK_SIZE; ++k) {	
                        if (a-aBegin+k<K0) {
                            Csub += AS(ty, k) * BS(k, tx);
                        }
                    }         
                }
            }
            __syncthreads();
        }
    }
    
    // Write the block sub-matrix to device memory;
    // each thread writes one element
	int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}

__global__ void
Matrix_dot_row_inplace_byC(float *A, unsigned int *C, float* W, int N0, int T, int T0)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;

    int index = by * blockDim.y + ty;    
    int index2 = bx * blockDim.x + tx;    
    if (index < N0) {
        unsigned int idx = C[index];
        __syncthreads();
        
        float wv = W[idx * T +index2];
        float av = A[index * T + index2];
        float anv = wv * av;
        __syncthreads();
        A[index * T + index2] = anv;
        __syncthreads();
    }
}

/*
 * Thread-block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datum within block
 * blockIdx.x counts data block
 *
 */
__global__ void Normalize(
						REAL* inMeasure, /** Precomputed measure */
						int iN,
						int iT) {

	const int kTJ = iT;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
    const int Tpad = iAlignUp(kTJ,HALFWARP);
	const int pdfIndex = datumIndex * Tpad;
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	__shared__ REAL sum[SAMPLE_BLOCK];
	if (thidx == 0) {
        sum[thidy] = 0;
	}
	//get scaled cummulative pdfs
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    sum[thidy] += measure[thidy][i];
                }
			}
		}
		__syncthreads();
	}
    
    REAL dv;
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
        dv = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
        inMeasure[pdfIndex + chunk + thidx] = dv /sum[thidy];
        __syncthreads();
	}
}
__global__ void Normalize_log(
						REAL* inMeasure, /** Precomputed measure */
						int iN,
						int iT) {

	const int kTJ = iT;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
    const int Tpad = iAlignUp(kTJ,HALFWARP);
	const int pdfIndex = datumIndex * Tpad;
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	__shared__ REAL sum[SAMPLE_BLOCK];
	if (thidx == 0) {
        sum[thidy] = 0;
	}
	//get scaled cummulative pdfs
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    sum[thidy] += measure[thidy][i];
                }
			}
		}
		__syncthreads();
	}
    
    REAL dv;
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
        dv = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
        inMeasure[pdfIndex + chunk + thidx] = log(dv /sum[thidy]);
        __syncthreads();
	}
}


/*
 * Thread-block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datum within block
 * blockIdx.x counts data block
 *
 */
__global__ void SumLog_K(
						REAL* inMeasure, /** Precomputed measure */
						int iN,
						int iT,
                        REAL* dR
                        ) {

	const int kTJ = iT;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
    const int Tpad = iAlignUp(kTJ,HALFWARP);
	const int pdfIndex = datumIndex * Tpad;
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	__shared__ REAL sum[SAMPLE_BLOCK];
	if (thidx == 0) {
        sum[thidy] = 0;
	}
	//get scaled cummulative pdfs
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    sum[thidy] += measure[thidy][i];
                }
			}
		}
		__syncthreads();
	}    
    if (thidx==0) {
        dR[datumIndex] = log(sum[thidy]);
        __syncthreads();
    }
}

/*
 * Thread-block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datum within block
 * blockIdx.x counts data block
 *
 */
__global__ void GetMaxFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						int* outComponent, /** Resultant choice */
						int iN,
						int iT
					) {
	const int kTJ = iT;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
    const int Tpad = iAlignUp(kTJ,HALFWARP);
	const int pdfIndex = datumIndex * Tpad;
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	int index = -1;
    REAL maxpdf = -1000.0;
    
    for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
        if (chunk + thidx < kTJ) { 
            measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
        }
        __syncthreads();
        if (thidx == 0) {
            for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    REAL dcurrent = measure[thidy][i];
                    if (dcurrent > maxpdf) {
                        maxpdf = dcurrent;
                        index = chunk + i;
                    }
                }
            }
        }
        __syncthreads();
    }
    
	if (thidx == 0 && datumIndex < iN) {
		outComponent[datumIndex] = index;
    }
}
/*
 * Thread-block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datum within block
 * blockIdx.x counts data block
 *
 */
__global__ void sampleFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						int* outComponent, /** Resultant choice */
						int iN,
						int iT,
                        int isLogScaled
					) {

	const int kTJ = iT;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
    const int Tpad = iAlignUp(kTJ,HALFWARP);
	const int pdfIndex = datumIndex * Tpad;
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	__shared__ REAL sum[SAMPLE_BLOCK];
	__shared__ REAL maxpdf[SAMPLE_BLOCK]; //only used when log scaled

	if (thidx == 0) {
        if (isLogScaled > 0) {
            maxpdf[thidy] = -1000.0;
        } 
        sum[thidy] = 0;
	}
    if (isLogScaled > 0) {
        //first scan: get the max values
        for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
            if (chunk + thidx < kTJ) { 
                measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
            }
            __syncthreads();
            if (thidx == 0) {
                for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                    if (chunk + i < kTJ) {
                        REAL dcurrent = measure[thidy][i];
                        if (dcurrent > maxpdf[thidy]) {
                            maxpdf[thidy] = dcurrent;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
	//second scan: get scaled cummulative pdfs
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    if (isLogScaled > 0) {
                        sum[thidy] += exp(measure[thidy][i]-maxpdf[thidy]);		//rescale and exp()
                    } else {
                        sum[thidy] += measure[thidy][i];
                    }
                }
				measure[thidy][i] = sum[thidy];
			}
		}
		if (chunk + thidx < kTJ) 
			inMeasure[pdfIndex + chunk + thidx] = measure[thidy][thidx];
		__syncthreads();
	}
	REAL randomNumber = inRandomNumber[datumIndex] * sum[thidy];
	int index = 0;
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		if (chunk + thidx < kTJ) 
			measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
                if (chunk + i < kTJ) {
                    if (randomNumber > measure[thidy][i]) {
                        index = i + chunk + 1;
                    }
                    if (index ==kTJ) {index = kTJ-1;}
                }
			}
		}
	}
	if (thidx == 0 && datumIndex < iN) 
		outComponent[datumIndex] = index;
}

//define this only if DENSITIES_IN_BLOCK and DATA_IN_BLOCK are both 16
//TWISTED_DENSITY should only be defined when iTJ is a multiple of 16
//#define TWISTED_DENSITY        
__global__ void mvNormalPDF(
					REAL* inData, /** Data-vector; padded */
					REAL* inDensityInfo, /** Density info; already padded */
					REAL* outPDF, /** Resultant PDF */
					int iD, 
					int iN,
					int iTJ,
					int isLogScaled
				) {
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int dataBlockIndex = blockIdx.x * DATA_IN_BLOCK;
	const int datumIndex = dataBlockIndex + thidx;
	const int densityBlockIndex = blockIdx.y * DENSITIES_IN_BLOCK;
	const int densityIndex = densityBlockIndex + thidy;
    
    int LOGDET_OFFSET = iD * (iD + 3) / 2;
	int MEAN_CHD_DIM = iD * (iD + 3) / 2	+ 2;	
	int PACKpad = iAlignUp(MEAN_CHD_DIM,HALFWARP);
	int Dpad = iAlignUp(iD,HALFWARP);
    int Tpad = iAlignUp(iTJ,HALFWARP);


	#if defined(TWISTED_DENSITY)
		const int pdfIndex = blockIdx.x * DATA_IN_BLOCK * Tpad + 
			blockIdx.y * DENSITIES_IN_BLOCK + thidy * Tpad + thidx;
	#else
		const int pdfIndex = datumIndex * Tpad + densityIndex;
	#endif
	extern __shared__ REAL sData[];
	REAL *densityInfo = sData;
    
	
	const int data_offset = DENSITIES_IN_BLOCK * PACKpad;
	REAL *data = &sData[data_offset];
	#if defined(TWISTED_DENSITY)
		REAL *result_trans = &sData[data_offset+DATA_IN_BLOCK * iD];
	#endif
	//Read in data
	for(int chunk = 0; chunk < iD; chunk += DENSITIES_IN_BLOCK) {
        if (chunk + thidy < iD ) {
            data[thidx * iD + chunk + thidy] = inData[Dpad*datumIndex + chunk + thidy];
        }
    }
	// Read in density info by chunks
	for(int chunk = 0; chunk < PACKpad; chunk += DATA_IN_BLOCK) {
		if (chunk + thidx < PACKpad) {
            if (densityIndex < iTJ) {
                densityInfo[thidy * PACKpad + chunk + thidx] = inDensityInfo[PACKpad*densityIndex	+ chunk + thidx];
            }
		}
	}
	__syncthreads();
    if (datumIndex < iN & densityIndex < iTJ) {
        // Setup pointers
        REAL* tData = data+thidx*iD;
        REAL* tDensityInfo = densityInfo + thidy * PACKpad;
        REAL* tMean = tDensityInfo;			
        REAL* tSigma = tDensityInfo + iD;
        REAL  tP = tDensityInfo[LOGDET_OFFSET];
        REAL  tLogDet = tDensityInfo[LOGDET_OFFSET+1];
        // Do density calculation
        REAL discrim = 0;
        for(int i=0; i<iD; i++) {
            REAL sum = 0;
            for(int j=0; j<=i; j++) {
                sum += *tSigma++ * (tData[j] - tMean[j]); // xx[j] is always calculated since j <= i
            }
            discrim += sum * sum;
        }
        REAL d;
        //REAL mydim = (REAL)iD;
        if (isLogScaled>0) {
            //d = log(tP)-0.5 * (discrim + tLogDet +(LOG_2_PI*mydim));
            d = log(tP)-0.5 * (discrim + tLogDet);
        } else {
            //d = tP * exp(-0.5 * (discrim + tLogDet + (LOG_2_PI*mydim))); 
            d = tP * exp(-0.5 * (discrim + tLogDet)); 
        }
        #if defined(TWISTED_DENSITY)
            result_trans[thidx * DATA_IN_BLOCK + thidy] = d;	
            __syncthreads();
        #endif
		#if defined(TWISTED_DENSITY)
			outPDF[pdfIndex] = result_trans[thidx + thidy * DENSITIES_IN_BLOCK];
		#else
			outPDF[pdfIndex] = d;
		#endif
	}

}