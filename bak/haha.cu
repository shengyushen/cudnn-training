/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "readubyte.h"

///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities

// Block width for CUDA kernels
#define BW 128

#ifdef USE_GFLAGS
#include <gflags/gflags.h>

#ifndef _WIN32
#define gflags google
#endif
#else
    // Constant versions of gflags
#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif


//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

__global__
void ssyinitfloat (float *p, size_t n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t idx = index; idx < n; idx += stride)
    {
      p[idx] = 0.0;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
// Main function
//#define WIDTH 280
int
main (int argc, char **argv)
{
#ifdef USE_GFLAGS
  gflags::ParseCommandLineFlags (&argc, &argv, true);
#endif
	if(argc!=4) {
		printf("Usage : haha.exe width fract iterations\n");
		assert(0);
	}
	size_t width=atoi(argv[1]);
	size_t sz=width*width;
	float fract = atof(argv[2]);
	int iterations = atoi(argv[3]);
	printf("size is %d\n",sz);

  // Choose GPU
  int num_gpus;
  checkCudaErrors (cudaGetDeviceCount (&num_gpus));
  printf ("using %d GPUs \n", num_gpus);

  int threadsPerBlock = 256;
  int numberOfBlocks = 32*80 ;

  std::vector < float *>d_dataV;


	//alloc on the gpus
  for (int gpuid = 0; gpuid < num_gpus; gpuid++) {
      float *d_data;
			checkCudaErrors (cudaSetDevice (gpuid));
      checkCudaErrors (cudaMallocManaged (&d_data, sizeof (float)*sz )); 
			checkCudaErrors (cudaMemAdvise(d_data,sizeof(float)*sz,cudaMemAdviseSetPreferredLocation,gpuid));
			d_dataV.push_back (d_data);
	}
	//init on the gpus
	for (int gpuid = 0; gpuid < num_gpus; gpuid++) {
			checkCudaErrors (cudaSetDevice (gpuid));
      ssyinitfloat <<< numberOfBlocks, threadsPerBlock >>> (d_dataV[gpuid], sz);
	}

  auto t1 = std::chrono::high_resolution_clock::now ();
  for (int iter = 0; iter < iterations; ++iter)
  {
		//copy
	  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
	  {
	      checkCudaErrors (cudaSetDevice (gpuid));
				if(gpuid>0) {
//				  checkCudaErrors (cudaMemPrefetchAsync (d_dataV[gpuid-1], int (fract * sz*sizeof(float) / 2), gpuid));
	  	    //on gpu n, copy  data from gpu n-1 to gpu n
				  checkCudaErrors (cudaMemcpyAsync (d_dataV[gpuid] + sz /2, d_dataV[gpuid - 1], int (fract * sz*sizeof(float) / 2), cudaMemcpyDefault));
				}
	  }

		//sync
	  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
	  {
	      checkCudaErrors (cudaSetDevice (gpuid));
	      checkCudaErrors (cudaDeviceSynchronize ());
	  }

		//vitis
	  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
		{
		  checkCudaErrors (cudaSetDevice (gpuid)); 
			ssyinitfloat <<< numberOfBlocks, threadsPerBlock >>> (d_dataV[gpuid], sz);
		}

	}

  checkCudaErrors (cudaDeviceSynchronize ());
  auto t2 = std::chrono::high_resolution_clock::now ();

  printf ("Iteration time: size %d fract %f time %f ms\n", sz,
	  fract ,
	  std::chrono::duration_cast < std::chrono::microseconds >
	  (t2 - t1).count () / 1000.0f / iterations);

  // Free data structures
  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
  {
      checkCudaErrors (cudaFree (d_dataV[gpuid]));
  }
  return 0;
}
