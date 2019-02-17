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

using namespace std;

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



//seems to be a class like struct
class baseModule{
	public : 
	string name;
  int in_channels, out_channels ;
  int in_width, in_height, out_width, out_height;

	float * pin; // this is pass in from outside
	long inbuf_size;
	float * pout; // pout should be alloc by child class
	long outbuf_size;

	int minibatch;

	int gpuid;

  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;

	size_t m_workspaceSizeByte;
	void * p_workspace;

	bool bNeedSyncInTensor;


	void print() {
					cout<<"name "<<name<<endl;
					cout<<"in_channels "<<in_channels<<endl;
					cout<<"out_channels "<<out_channels<<endl;
					cout<<"in_width "<<in_width<<endl;
					cout<<"in_height "<<in_height<<endl;
					cout<<"out_width "<<out_width<<endl;
					cout<<"out_height "<<out_height<<endl;
					cout<<"inbuf_size "<<inbuf_size<<endl;
					cout<<"outbuf_size "<<outbuf_size<<endl;
					cout<<"minibatch "<<minibatch<<endl;
					cout<<"gpuid "<<gpuid<<endl;
	}
	baseModule(
									string name_,
  								cudnnHandle_t cudnnHandle_,
								  cublasHandle_t cublasHandle_,
									int gpuid_,
									int minibatch_,
									int in_c_,
									int out_c_,
									int in_h_,
									int in_w_,
									int out_h_,
									int out_w_,
									float * pin_
									) 
	{
		name=name_;
		cudnnHandle = cudnnHandle_;
		cublasHandle = cublasHandle_;
		gpuid = gpuid_;
		minibatch = minibatch_;
		in_channels = in_c_;
		out_channels = out_c_;
		in_width = in_w_;
		in_height = in_h_;
		out_width = out_w_;
		out_height = out_h_;
		pin = pin_;
		inbuf_size = minibatch_*in_c_*in_w_*in_h_;
		outbuf_size = minibatch_*out_c_*out_w_*out_h_;
		m_workspaceSizeByte=0;
		p_workspace=NULL;
		bNeedSyncInTensor=true;

		assert(gpuid>=0);
		assert(minibatch >0);
		assert(in_channels >0);
		assert(out_channels >0);
		assert(in_width >0);
		assert(in_height >0);
		assert(out_width >0);
		assert(out_height >0);
		assert(pin );

		checkCudaErrors(cudaSetDevice(gpuid));
    checkCudaErrors (cudaMallocManaged (&pout, sizeof (float) *outbuf_size ));
    checkCudaErrors (cudaMemAdvise (pout,sizeof(float)* outbuf_size ,cudaMemAdviseSetPreferredLocation,gpuid));
	}

	virtual void run1step() {};

	~baseModule  () {
			checkCudaErrors(cudaSetDevice(gpuid));
			cudaFree(pout);
	}
	size_t getOutputFloatNumber() {
					return outbuf_size;
	}
	size_t getInputFloatNumber() {
					return inbuf_size;
	}

};

class ConvBiasLayer: public baseModule
{
		public :
		int kernel_size,stride;
		cudnnTensorDescriptor_t biasTensor;
		float * pconvbias;
		cudnnTensorDescriptor_t srcTensorDesc;
		cudnnFilterDescriptor_t filterDesc;
		float * pconvWeigth;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnTensorDescriptor_t  dstTensorDesc; //this out already have pout in baseModule
		cudnnConvolutionFwdAlgo_t algo;

    ConvBiasLayer (
				string name_,
				cudnnHandle_t cudnnHandle_,
			  cublasHandle_t cublasHandle_,
				int gpuid_,
				int minibatch_,
				int in_channels_, 
				int in_h_, int in_w_, 
				int numFilter_, int kernel_size_, int stride_, 
				int paddingH_, int paddingW_,
				float * pin_)  // pin pass from outside
						: baseModule(
									name_,
									cudnnHandle_,
									cublasHandle_,
									gpuid_,
									minibatch_,
									in_channels_,
									numFilter_,
									in_h_,
									in_w_,
									(in_h_+paddingH_*2-kernel_size_+1)/stride_,
									(in_w_+paddingW_*2-kernel_size_+1)/stride_,
									pin_
								)
		{
						printf("ConvBiasLayer gpuid %d minibatch %d in_channels_ %d in_h_ %d in_w_ %d numFilter_ %d kernel_size_ %d stride_ %d paddingH_ %d paddingW_ %d\n",
						                      gpuid ,  minibatch ,  in_channels_ ,  in_h_ ,  in_w_ ,  numFilter_ ,  kernel_size_ ,  stride_ ,  paddingH_ ,  paddingW_ );
				assert((in_w_+paddingW_*2-kernel_size_+1)%stride_ == 0);
				assert((in_h_+paddingH_*2-kernel_size_+1)%stride_ == 0);

				kernel_size = kernel_size_;
				assert(kernel_size<16); //this is not strict, just to prevent unreasonable large kernel
				stride=stride_;
				assert(stride < 16);//also not strict

				//bias descriptor
				checkCUDNN (cudnnCreateTensorDescriptor (&biasTensor));
    		checkCUDNN (cudnnSetTensor4dDescriptor (biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));
    		checkCudaErrors(cudaMallocManaged(&pconvbias, sizeof(float) * out_channels ));

				//set the source tensor
    		checkCUDNN (cudnnCreateTensorDescriptor (&srcTensorDesc));
				//this may fail in 800 pixel because the cudnnSetTensor4dDescriptor require the tensor smaller than 2GB, so I may need 700
    		checkCUDNN (cudnnSetTensor4dDescriptor (srcTensorDesc,
					    CUDNN_TENSOR_NCHW,
					    CUDNN_DATA_FLOAT, minibatch, in_channels, in_height, in_width));

				//set the filter desc
    		checkCUDNN (cudnnCreateFilterDescriptor (&filterDesc));
    		checkCUDNN (cudnnSetFilter4dDescriptor (filterDesc,
					    CUDNN_DATA_FLOAT,
					    CUDNN_TENSOR_NCHW,
					    out_channels,
					    in_channels,
					    kernel_size,
					    kernel_size));
				checkCudaErrors(cudaMallocManaged(&pconvWeigth,sizeof(float)*in_channels_*kernel_size_*kernel_size_*numFilter_));

    		checkCUDNN (cudnnCreateConvolutionDescriptor (&convDesc));
    		checkCUDNN (cudnnSetConvolution2dDescriptor (convDesc,
						 paddingH_, paddingW_,
						 stride, stride,
						 1, 1, // we currently dont support dilation
						 CUDNN_CROSS_CORRELATION,
						 CUDNN_DATA_FLOAT));

				int n,c,h,w;
    		checkCUDNN (cudnnGetConvolution2dForwardOutputDim (convDesc,
						       srcTensorDesc,
						       filterDesc,
						       &n, &c, &h, &w));
				assert(n==minibatch);
				assert(c=out_channels);
				assert(h==out_height);
				assert(w==out_width);

    		checkCUDNN (cudnnCreateTensorDescriptor (&dstTensorDesc));
    		checkCUDNN (cudnnSetTensor4dDescriptor (dstTensorDesc,
					    CUDNN_TENSOR_NCHW,
					    CUDNN_DATA_FLOAT, n, c, h, w));
		    checkCUDNN (cudnnGetConvolutionForwardAlgorithm (cudnnHandle,
						     srcTensorDesc,
						     filterDesc,
						     convDesc,
						     dstTensorDesc,
						     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
						     0, &algo));

		    checkCUDNN (cudnnGetConvolutionForwardWorkspaceSize (cudnnHandle,
							 srcTensorDesc,
							 filterDesc,
							 convDesc,
							 dstTensorDesc,
							 algo, &m_workspaceSizeByte));
		}
	void run1step () {
				assert(p_workspace!=NULL);
				assert(m_workspaceSizeByte!=0);
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(gpuid));
				assert(p_workspace);
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, 
																					&alpha, 
																					srcTensorDesc, pin, 
																					filterDesc, pconvWeigth, 
																					convDesc, 
                                           algo, p_workspace, m_workspaceSizeByte, &beta,
                                           dstTensorDesc, pout));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensor,
                                  pconvbias, &alpha, dstTensorDesc, pout));

	}
	~ConvBiasLayer() {
		checkCUDNN (cudnnDestroyTensorDescriptor(biasTensor));
		checkCUDNN (cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN (cudnnDestroyFilterDescriptor(filterDesc));
		checkCudaErrors(cudaFree(pconvWeigth));
		checkCUDNN (cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN (cudnnDestroyTensorDescriptor (dstTensorDesc));
	}

};


///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void
FillOnes (float *vec, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  vec[idx] = 1.0f;
}

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void
SoftmaxLossBackprop (const float *label, int num_labels, int batch_size,
		     float *diff)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  const int label_value = static_cast < int >(label[idx]);

  // For each item in the batch, decrease the result of the label's value by 1
  diff[idx * num_labels + label_value] -= 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

class TrainingContext
{
	public :

  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
  int m_gpuid;
  int m_batchSize;
	//only the first tensor is need to store here
  cudnnTensorDescriptor_t dataTensor;

	std::vector<class baseModule *> vmod;
	int currentlayer;

	void * pworkspace;

	void print() {
					printf("TrainingContext m_gpuid %d m_batchSize %d\n",m_gpuid,m_batchSize);
					for(int i =0;i<vmod.size();i++) {
						vmod[i]->print();
					}
	}

  TrainingContext (int gpuid, int batch_size)
  {
    m_batchSize = batch_size;
		m_gpuid =gpuid;
    printf ("gpuid %d batch_size %d\n", gpuid,batch_size);
		currentlayer=0;

    // Create CUBLAS and CUDNN handles
    checkCudaErrors (cudaSetDevice (gpuid));
    checkCudaErrors (cublasCreate (&cublasHandle));
    checkCUDNN (cudnnCreate (&cudnnHandle));

    // Create tensor descriptors
    checkCUDNN (cudnnCreateTensorDescriptor (&dataTensor));
  }

	void addMod(class baseModule * pmod) {
		vmod.push_back(pmod);
	}

	baseModule * getCurrentLayer() {
			assert(currentlayer >=0);
			if(currentlayer >= vmod.size()) {
					cout<<"currentlayer "<<currentlayer<<"vmod.size "<<vmod.size()<<endl;
					assert(0);
			}
			return vmod[currentlayer];
	}

	void reset() {
					currentlayer=0;
	}

	bool isFinished() {
		 if(currentlayer>=vmod.size()) return true;
		 else return false;
	}

	void finishAddMod () {
		size_t maxsize=0;
		for(int i=0;i<vmod.size();i++) {
			maxsize = std::max(maxsize,vmod[i]->m_workspaceSizeByte);
		}
		//alloc new size
		checkCudaErrors(cudaMallocManaged(&pworkspace,maxsize));
		for(int i=0;i<vmod.size();i++) {
				vmod[i]->p_workspace = pworkspace;
				vmod[i]->m_workspaceSizeByte=maxsize;
		}
	}

   ~TrainingContext ()
  {
		for(int i=0;i<vmod.size();i++) {
			delete vmod[i];
		}
    checkCudaErrors (cudaSetDevice (m_gpuid));

    checkCUDNN (cudnnDestroyTensorDescriptor (dataTensor));
    checkCudaErrors (cublasDestroy (cublasHandle));
    checkCUDNN (cudnnDestroy (cudnnHandle));
  }

	 void ForwardPropagation1() {
		 if(currentlayer>=vmod.size()) {
 		  cout<<"finished at layer "<<currentlayer<<endl;
			assert(0);
		 } else {
				cout<<"layer "<<currentlayer<<endl;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1 layer
				vmod[currentlayer]->run1step();
				
				currentlayer++;
		 }
	 }
};

__global__ void
ssyinitfloat (float *p, size_t n)
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
  if (argc != 7) {
		printf("Usage : cudnnModelParallel.exe <width> <iteration> <minbatch> <channel> <copy or not> <fract to copy>");
		assert(0);
	}
	cout<<"argc "<<argc<<endl;
  size_t width, height;
  width = atoi (argv[1]);
  height = width;
	cout<<"width "<<width<<endl;
  int iterations = atoi (argv[2]);
	int minib = atoi(argv[3]);
	int chnl = atoi(argv[4]);
  bool copy = (atoi (argv[5]) > 0);
  float fract = (atof (argv[6]));

  // Choose GPU
  int num_gpus;
  checkCudaErrors (cudaGetDeviceCount (&num_gpus));
	cout<<"num_gpus "<<num_gpus<<endl;

	int deviceId;
//  int numberOfSMs;
	checkCudaErrors(cudaSetDevice(0));
	cudaGetDevice(&deviceId);
	cout<<"deviceId "<<deviceId<<endl;
//	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//printf("numberOfSMs %s\n",numberOfSMs);
//  int threadsPerBlock = 256;
//  int numberOfBlocks = 32*80 ;


  std::vector < float *>d_dataV;
	std::vector <TrainingContext * >contextV;

  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
    {
      checkCudaErrors (cudaSetDevice (gpuid));
			//alloc the input data
			float * pdata;
			size_t input_sz = minib*chnl*width*height;
			checkCudaErrors(cudaMallocManaged(&pdata,sizeof(float)*input_sz));
			//the context for this gpu
      TrainingContext * pcontext = new TrainingContext (gpuid, minib);

      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											"conv1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											chnl,
											height,width, 
											64,1,1,
											0,0,
											pdata
											);
			class ConvBiasLayer * pconv2=new ConvBiasLayer (
											"conv2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pconv1->out_channels,
											pconv1->out_height, pconv1->out_width,
											64,3,1,
											1,1,
											pconv1->pout
											);
			class ConvBiasLayer * pconv3=new ConvBiasLayer (
											"conv3",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pconv2->out_channels,
											pconv2->out_height, pconv2->out_width,
											256,1,1,
											0,0,
											pconv2->pout
											);

			pcontext -> addMod(pconv1);
			pcontext -> addMod(pconv2);
			pcontext -> addMod(pconv3);
			pcontext -> finishAddMod();

      contextV.push_back (pcontext);
			d_dataV.push_back(pdata);
	}

	for (int gpuid = 0; gpuid < num_gpus; gpuid++) {
					contextV[gpuid]-> print();
	}

  checkCudaErrors (cudaDeviceSynchronize ());

  // Use SGD to train the network
  auto t1 = std::chrono::high_resolution_clock::now ();
  for (int iter = 0; iter < iterations; ++iter)
  {
		//reset
		for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
						contextV[gpuid]->reset();
		}

		while(true) {
			//run one layer
		  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			{
				assert(contextV[gpuid]->m_gpuid == gpuid);
			  checkCudaErrors (cudaSetDevice (gpuid));
			  contextV[gpuid]->ForwardPropagation1 ();
			}
			if(contextV[0]->isFinished()) break;
			
		  if (copy)
			{
			  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			  {
			      //sync n+1 to n
			      checkCudaErrors (cudaSetDevice (gpuid));
						baseModule * pcurrent =contextV[gpuid]->getCurrentLayer();
			      size_t sz = sizeof (float) * (pcurrent->getInputFloatNumber() );
						assert(sz>0);
						cout<<"sz "<<sz<<endl;

			      if (gpuid > 0) {
							baseModule * pPrev =contextV[gpuid-1]->getCurrentLayer();
							size_t szPrev = sizeof (float) * (pPrev->getInputFloatNumber() );
							assert(sz==szPrev);
						  checkCudaErrors (cudaMemcpyAsync (pcurrent->pin + sz / (2 * sizeof (float)), pPrev->pin, int (fract * sz / 2), cudaMemcpyDefault));
						}
			  }
		
			  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			    {
			      checkCudaErrors (cudaSetDevice (gpuid));
			      checkCudaErrors (cudaDeviceSynchronize ());
			    }
			}
		}
  }				// end of iteration

  checkCudaErrors (cudaDeviceSynchronize ());
  auto t2 = std::chrono::high_resolution_clock::now ();

  cout<<"Iteration time: width "<<width
			<<" fract "<<
			(copy?fract:0.0)
			<<" time " << std::chrono::duration_cast < std::chrono::microseconds > (t2 - t1).count () / 1000.0f / iterations
		<<" ms"<<endl;
  return 0;
}
