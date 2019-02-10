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

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

/**
 * Saves a PGM grayscale image out of unsigned 8-bit data
 */
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
        fwrite(data, sizeof(unsigned char), width * height, fp);
        fclose(fp);
    }
}

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


///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size, 64, "Batch size for training");

// Filenames
DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");
DEFINE_string(train_images, "train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels-idx1-ubyte", "Test labels filename");

// Solver parameters
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");


///////////////////////////////////////////////////////////////////////////////////////////
// Layer representations

/**
 * Represents a convolutional layer with bias.
 */
//seems to be a class like struct
struct ConvBiasLayer
{
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;
    
    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, 
                  int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), 
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";
        
        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    }
};

/**
 * Represents a max-pooling layer.
 */
struct MaxPoolLayer
{
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

/**
 * Represents a fully-connected neural network layer with bias.
 */
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
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
__global__ void FillOnes(float *vec, int size)
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
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor, 
                             conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer* ref_pfc1, *ref_pfc2;

    // Disable copying
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;
	//ssy should need move to enable std::vector
    //TrainingContext& operator=(TrainingContext&&) noexcept {}
    //TrainingContext(TrainingContext&&) noexcept {}

    TrainingContext(int gpuid, int batch_size,
                    ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                    FullyConnectedLayer* pfc1, FullyConnectedLayer* pfc2) : ref_pfc1(pfc1), ref_pfc2(pfc2), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;
	printf("ref_pfc1->inputs %d\n",ref_pfc1->inputs);
	printf("ref_pfc1->outputs %d\n",ref_pfc1->outputs);
	printf("gpuid %d\n",gpuid);

        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

        checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));            

        
        // Set tensor descriptor sizes
        checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv1.out_channels,
                                              1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              1, conv2.out_channels,
                                              1, 1));
            
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                               CUDNN_POOLING_MAX,
                                               CUDNN_PROPAGATE_NAN,
                                               pool1.size, pool1.size,
                                               0, 0,
                                               pool1.stride, pool1.stride));
        checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, conv2.out_channels,
                                              conv2.out_height / pool2.stride,
                                              conv2.out_width / pool2.stride));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, pfc1->outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, pfc2->outputs, 1, 1));

        checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN, 0.0));


        // Set convolution tensor sizes and compute workspace size
        size_t workspace = 0;
        workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

        workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

        // The workspace is allocated later (if necessary)
        m_workspaceSize = workspace;
    }

    ~TrainingContext()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    }

    size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionFwdAlgo_t& algo)
    {
        size_t sizeInBytes = 0;

        int n = m_batchSize;
        int c = conv.in_channels;
        int h = conv.in_height;
        int w = conv.in_width;
//ssy: the size of each tensor is set here
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW,
                                              conv.out_channels,
                                              conv.in_channels, 
                                              conv.kernel_size,
                                              conv.kernel_size));

#if CUDNN_MAJOR > 5
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));
#else
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   CUDNN_CROSS_CORRELATION));
#endif

        // Find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         &n, &c, &h, &w));

        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &algo));
        
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           srcTensorDesc,
                                                           filterDesc,
                                                           convDesc,
                                                           dstTensorDesc,
                                                           algo,
                                                           &sizeInBytes));

        return sizeInBytes;
    }

    void ForwardPropagation1(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias, 
                            float *pconv2, float *pconv2bias, 
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));
	printf("%s m_gpuid= %d\n",__FUNCTION__,m_gpuid);

        // Conv1 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                           data, conv1filterDesc, pconv1, conv1Desc, 
                                           conv1algo, workspace, m_workspaceSize, &beta,
                                           conv1Tensor, conv1));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                                  pconv1bias, &alpha, conv1Tensor, conv1));

        // Pool1 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                       conv1, &beta, pool1Tensor, pool1));

    }

    void ForwardPropagation2(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias, 
                            float *pconv2, float *pconv2bias, 
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));
	printf("%s m_gpuid= %d\n",__FUNCTION__,m_gpuid);

        // Conv2 layer
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                           pool1, conv2filterDesc, pconv2, conv2Desc, 
                                           conv2algo, workspace, m_workspaceSize, &beta,
                                           conv2Tensor, conv2));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                  pconv2bias, &alpha, conv2Tensor, conv2));

        // Pool2 layer
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                       conv2, &beta, pool2Tensor, pool2));
    }

    void ForwardPropagation3(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias, 
                            float *pconv2, float *pconv2bias, 
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));
	printf("%s m_gpuid= %d\n",__FUNCTION__,m_gpuid);


        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
//	printf("ref_pfc1->outputs %d\n",ref_pfc1->outputs);
//	printf("ref_pfc1->inputs %d\n",ref_pfc1->inputs);
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_pfc1->outputs, m_batchSize, ref_pfc1->inputs,
                                    &alpha,
                                    pfc1, ref_pfc1->inputs,
                                    pool2, ref_pfc1->inputs,
                                    &beta,
                                    fc1, ref_pfc1->outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_pfc1->outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc1bias, ref_pfc1->outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_pfc1->outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                          fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_pfc2->outputs, m_batchSize, ref_pfc2->inputs,
                                    &alpha,
                                    pfc2, ref_pfc2->inputs,
                                    fc1relu, ref_pfc2->inputs,
                                    &beta,
                                    fc2, ref_pfc2->outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_pfc2->outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc2bias, ref_pfc2->outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_pfc2->outputs));

        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, 
                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
    {
        size_t sizeInBytes = 0, tmpsize = 0;

        // If backprop filter algorithm was requested
        if (falgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
                *falgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        // If backprop data algorithm was requested
        if (dalgo)
        {
            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
                *dalgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }
        
        return sizeInBytes;
    }

    void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                         float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                         float *fc2, float *fc2smax, float *dloss_data,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2,
                         void *workspace, float *onevec)
    {    
        float alpha = 1.0f, beta = 0.0f;

        float scalVal = 1.0f / static_cast<float>(m_batchSize);

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_pfc2->outputs, cudaMemcpyDeviceToDevice));
        
        // Softmax layer
        SoftmaxLossBackprop<<<RoundUp(m_batchSize, BW), BW>>>(labels, ref_pfc2->outputs, m_batchSize, dloss_data);

        // Accounting for batch size in SGD
        checkCudaErrors(cublasSscal(cublasHandle, ref_pfc2->outputs * m_batchSize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_pfc2->inputs, ref_pfc2->outputs, m_batchSize,
                                    &alpha, fc1relu, ref_pfc2->inputs, dloss_data, ref_pfc2->outputs, &beta, gfc2, ref_pfc2->inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_pfc2->outputs, m_batchSize,
                                    &alpha, dloss_data, ref_pfc2->outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_pfc2->inputs, m_batchSize, ref_pfc2->outputs,
                                    &alpha, pfc2, ref_pfc2->inputs, dloss_data, ref_pfc2->outputs, &beta, dfc2, ref_pfc2->inputs));
        
        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                           fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                           fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_pfc1->inputs, ref_pfc1->outputs, m_batchSize,
                                    &alpha, pool2, ref_pfc1->inputs, dfc1relu, ref_pfc1->outputs, &beta, gfc1, ref_pfc1->inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_pfc1->outputs, m_batchSize,
                                    &alpha, dfc1relu, ref_pfc1->outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_pfc1->inputs, m_batchSize, ref_pfc1->outputs,
                                    &alpha, pfc1, ref_pfc1->inputs, dfc1relu, ref_pfc1->outputs, &beta, dfc1, ref_pfc1->inputs));

        // Pool2 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        pool2Tensor, pool2, pool2Tensor, dfc1,
                                        conv2Tensor, conv2, &beta, conv2Tensor, dpool2));
        
        // Conv2 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                                dpool2, &beta, conv2BiasTensor, gconv2bias));

        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                                  pool1, conv2Tensor, dpool2, conv2Desc,
                                                  conv2bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv2filterDesc, gconv2));
    
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                                pconv2, conv2Tensor, dpool2, conv2Desc, 
                                                conv2bwdalgo, workspace, m_workspaceSize,
                                                &beta, pool1Tensor, dconv2));
        
        // Pool1 layer
        checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        pool1Tensor, pool1, pool1Tensor, dconv2,
                                        conv1Tensor, conv1, &beta, conv1Tensor, dpool1));
        
        // Conv1 layer
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                                dpool1, &beta, conv1BiasTensor, gconv1bias));
        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                  data, conv1Tensor, dpool1, conv1Desc,
                                                  conv1bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv1filterDesc, gconv1));

        // No need for convBackwardData because there are no more layers below
    }

    void UpdateWeights(float learning_rate,
                       ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias)
    {    
        float alpha = -learning_rate;

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                    &alpha, gconv1, 1, pconv1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                    &alpha, gconv1bias, 1, pconv1bias, 1));

        // Conv2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                    &alpha, gconv2, 1, pconv2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                    &alpha, gconv2bias, 1, pconv2bias, 1));

        // Fully connected 1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_pfc1->pneurons.size()),
                                    &alpha, gfc1, 1, pfc1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_pfc1->pbias.size()),
                                    &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_pfc2->pneurons.size()),
                                    &alpha, gfc2, 1, pfc2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_pfc2->pbias.size()),
                                    &alpha, gfc2bias, 1, pfc2bias, 1));
    }
};


///////////////////////////////////////////////////////////////////////////////////////////
// Main function
//#define WIDTH 280
int main(int argc, char **argv)
{
#ifdef USE_GFLAGS
    gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif
	assert(argc==5);
	printf("argc %d\n",argc);
	bool copy=(atoi(argv[3])>0);
	float fract = (atof(argv[4]));
    size_t width, height, channels = 1;

    // Open input data
    printf("Reading input data\n");
    
	printf("%s\n",FLAGS_train_images.c_str());
	printf("%s\n",FLAGS_train_labels.c_str());
    // Read dataset sizes
	//ssy only find out ethe size ,dont really read the data
//    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
    size_t train_size = 60000;
	width = atoi(argv[1]);
	height = width;
	printf("width %d\n",width);
//    size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
    size_t test_size = 10000;
    if (train_size == 0)
        return 1;
    
	printf("train_size %d\n",train_size);
	printf("test_size %d\n",test_size);
//ssy the temp buf to hold the image
    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
//    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
//        return 2;
//    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
//        return 3;

	int iterations=atoi(argv[2]);
    printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, iterations);

    // This code snippet saves a random image and its label
    /*
    std::random_device rd_image;
    int random_image = rd_image() % train_size;
    std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
    SavePGMFile(&train_images[0] + random_image * width*height*channels, width, height, ss.str().c_str());
    */

    // Choose GPU
    int num_gpus;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               FLAGS_gpu, num_gpus);
        return 4;
    }
	printf("using %d GPUs \n",num_gpus);

    // Create the LeNet network architecture
    std::vector<ConvBiasLayer> conv1V;
    std::vector<MaxPoolLayer> pool1V;
    std::vector<ConvBiasLayer> conv2V;
    std::vector<MaxPoolLayer> pool2V;
    std::vector<FullyConnectedLayer*> fc1V;
    std::vector<FullyConnectedLayer*> fc2V;

    // Initialize CUDNN/CUBLAS training context
	std::vector<TrainingContext*> contextV;

std::vector<float*> d_dataV; 
std::vector<float*> d_labelsV; 
std::vector<float*> d_conv1V; 
std::vector<float*> d_pool1V; 
std::vector<float*> d_conv2V; 
std::vector<float*> d_pool2V; 
std::vector<float*> d_fc1V; 
std::vector<float*> d_fc1reluV; 
std::vector<float*> d_fc2V; 
std::vector<float*> d_fc2smaxV;
std::vector<float*> d_pconv1V; 
std::vector<float*> d_pconv1biasV; 
std::vector<float*> d_pconv2V; 
std::vector<float*> d_pconv2biasV;
std::vector<float*> d_pfc1V; 
std::vector<float*> d_pfc1biasV; 
std::vector<float*> d_pfc2V; 
std::vector<float*> d_pfc2biasV;
std::vector<float*> d_gconv1V; 
std::vector<float*> d_gconv1biasV; 
std::vector<float*> d_gconv2V; 
std::vector<float*> d_gconv2biasV;
std::vector<float*> d_gfc1V; 
std::vector<float*> d_gfc1biasV; 
std::vector<float*> d_gfc2V; 
std::vector<float*> d_gfc2biasV;

std::vector<float *> d_onevecV;
std::vector<void *> d_cudnn_workspaceV;    
for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
        checkCudaErrors(cudaSetDevice(gpuid));
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer* pfc1 = new FullyConnectedLayer((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 500);
    FullyConnectedLayer* pfc2 = new FullyConnectedLayer(pfc1->outputs, 10);
    TrainingContext* pcontext=new TrainingContext(gpuid, FLAGS_batch_size, conv1, pool1, conv2, pool2, pfc1, pfc2);
	conv1V.push_back(conv1);
	pool1V.push_back(pool1);
	conv2V.push_back(conv2);
	pool2V.push_back(pool2);
	fc1V.push_back(pfc1);
	fc2V.push_back(pfc2);

	contextV.push_back(pcontext);
    // Determine initial network structure
    bool bRet = true;
    if (FLAGS_pretrained)
    {
      bRet = conv1.FromFile("conv1");
      bRet &= conv2.FromFile("conv2");
      bRet &= pfc1->FromFile("ip1");
      bRet &= pfc2->FromFile("ip2");
    }
    if (!bRet || !FLAGS_pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (pfc1->inputs * pfc1->outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (pfc2->inputs * pfc2->outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : pfc1->pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : pfc1->pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : pfc2->pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : pfc2->pbias)
            iter = static_cast<float>(dfc2(gen));
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures    

    // Forward propagation data
float *	d_data; 
float *	d_labels; 
float *	d_conv1; 
float *	d_pool1; 
float *	d_conv2; 
float *	d_pool2; 
float *	d_fc1; 
float *	d_fc1relu; 
float *	d_fc2; 
float *	d_fc2smax;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkCudaErrors(cudaMallocManaged(&d_data,    sizeof(float) * pcontext->m_batchSize * channels           * height                            * width));
    checkCudaErrors(cudaMallocManaged(&d_labels,  sizeof(float) * pcontext->m_batchSize * 1                  * 1                                 * 1));
    checkCudaErrors(cudaMallocManaged(&d_conv1,   sizeof(float) * pcontext->m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkCudaErrors(cudaMallocManaged(&d_pool1,   sizeof(float) * pcontext->m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkCudaErrors(cudaMallocManaged(&d_conv2,   sizeof(float) * pcontext->m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkCudaErrors(cudaMallocManaged(&d_pool2,   sizeof(float) * pcontext->m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    checkCudaErrors(cudaMallocManaged(&d_fc1,     sizeof(float) * pcontext->m_batchSize * pfc1->outputs));    
    checkCudaErrors(cudaMallocManaged(&d_fc1relu, sizeof(float) * pcontext->m_batchSize * pfc1->outputs));
    checkCudaErrors(cudaMallocManaged(&d_fc2,     sizeof(float) * pcontext->m_batchSize * pfc2->outputs));
    checkCudaErrors(cudaMallocManaged(&d_fc2smax, sizeof(float) * pcontext->m_batchSize * pfc2->outputs));    

    // Network parameters
    float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
    
    checkCudaErrors(cudaMallocManaged(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMallocManaged(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMallocManaged(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMallocManaged(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMallocManaged(&d_pfc1,       sizeof(float) * pfc1->pneurons.size()));
    checkCudaErrors(cudaMallocManaged(&d_pfc1bias,   sizeof(float) * pfc1->pbias.size()));
    checkCudaErrors(cudaMallocManaged(&d_pfc2,       sizeof(float) * pfc2->pneurons.size()));
    checkCudaErrors(cudaMallocManaged(&d_pfc2bias,   sizeof(float) * pfc2->pbias.size()));    
    
    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
    
    checkCudaErrors(cudaMallocManaged(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    checkCudaErrors(cudaMallocManaged(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    checkCudaErrors(cudaMallocManaged(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
    checkCudaErrors(cudaMallocManaged(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    checkCudaErrors(cudaMallocManaged(&d_gfc1,       sizeof(float) * pfc1->pneurons.size()));
    checkCudaErrors(cudaMallocManaged(&d_gfc1bias,   sizeof(float) * pfc1->pbias.size()));    
    checkCudaErrors(cudaMallocManaged(&d_gfc2,       sizeof(float) * pfc2->pneurons.size()));
    checkCudaErrors(cudaMallocManaged(&d_gfc2bias,   sizeof(float) * pfc2->pbias.size()));
    
 d_dataV       .push_back(d_data);
 d_labelsV     .push_back(d_labels);
 d_conv1V      .push_back(d_conv1);
 d_pool1V      .push_back(d_pool1);
 d_conv2V      .push_back(d_conv2);
 d_pool2V      .push_back(d_pool2);
 d_fc1V        .push_back(d_fc1);
 d_fc1reluV    .push_back(d_fc1relu);
 d_fc2V        .push_back(d_fc2);
 d_fc2smaxV    .push_back(d_fc2smax);
 d_pconv1V     .push_back(d_pconv1);
 d_pconv1biasV .push_back(d_pconv1bias);  
 d_pconv2V     .push_back(d_pconv2);
 d_pconv2biasV .push_back(d_pconv2bias);
 d_pfc1V       .push_back(d_pfc1);
 d_pfc1biasV   .push_back(d_pfc1bias);
 d_pfc2V       .push_back(d_pfc2);
 d_pfc2biasV   .push_back(d_pfc2bias);
 d_gconv1V     .push_back(d_gconv1);
 d_gconv1biasV .push_back(d_gconv1bias);  
 d_gconv2V     .push_back(d_gconv2);
 d_gconv2biasV .push_back( d_gconv2bias);
 d_gfc1V       .push_back(d_gfc1);
 d_gfc1biasV   .push_back(d_gfc1bias);
 d_gfc2V       .push_back(d_gfc2);
 d_gfc2biasV   .push_back(d_gfc2bias);
    // Differentials w.r.t. data
//ssy use in bp
//    float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
//    //                         Buffer     | Element       | N                   | C                  | H                                 | W
//    //-----------------------------------------------------------------------------------------------------------------------------------------
//    checkCudaErrors(cudaMallocManaged(&d_dpool1,   sizeof(float) * pcontext->m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
//    checkCudaErrors(cudaMallocManaged(&d_dpool2,   sizeof(float) * pcontext->m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
//    checkCudaErrors(cudaMallocManaged(&d_dconv2,   sizeof(float) * pcontext->m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
//    checkCudaErrors(cudaMallocManaged(&d_dfc1,     sizeof(float) * pcontext->m_batchSize * pfc1->inputs));
//    checkCudaErrors(cudaMallocManaged(&d_dfc1relu, sizeof(float) * pcontext->m_batchSize * pfc1->outputs));
//    checkCudaErrors(cudaMallocManaged(&d_dfc2,     sizeof(float) * pcontext->m_batchSize * pfc2->inputs));
//    checkCudaErrors(cudaMallocManaged(&d_dfc2smax, sizeof(float) * pcontext->m_batchSize * pfc2->outputs));
//    checkCudaErrors(cudaMallocManaged(&d_dlossdata,sizeof(float) * pcontext->m_batchSize * pfc2->outputs));
    
    // Temporary buffers and workspaces
    float *d_onevec;
    checkCudaErrors(cudaMallocManaged(&d_onevec, sizeof(float)* pcontext->m_batchSize));
	d_onevecV.push_back(d_onevec);

    void *d_cudnn_workspace = nullptr;    
    if (pcontext->m_workspaceSize > 0)
        checkCudaErrors(cudaMallocManaged(&d_cudnn_workspace, pcontext->m_workspaceSize));    
	
	d_cudnn_workspaceV.push_back(d_cudnn_workspace);
    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
//    checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &pfc1->pneurons[0],      sizeof(float) * pfc1->pneurons.size(), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &pfc1->pbias[0],     sizeof(float) * pfc1->pbias.size(),    cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &pfc2->pneurons[0],      sizeof(float) * pfc2->pneurons.size(), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &pfc2->pbias[0],     sizeof(float) * pfc2->pbias.size(),    cudaMemcpyHostToDevice));
    
    // Fill one-vector with ones
    FillOnes<<<RoundUp(pcontext->m_batchSize, BW), BW>>>(d_onevec, pcontext->m_batchSize);

}    
    printf("Preparing dataset\n");
    // Normalize training set to be in [0,1]
//    std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
//    for (size_t i = 0; i < train_size * channels * width * height; ++i)
//        train_images_float[i] = (float)train_images[i] / 255.0f;
//    
//    for (size_t i = 0; i < train_size; ++i)
//        train_labels_float[i] = (float)train_labels[i];

    printf("Training...\n");
	printf("iteration %d\n",iterations);
	printf("batch size  %d\n",contextV[0]->m_batchSize);
	printf("input data dimension %d %d  %d %d\n",contextV[0]->m_batchSize,channels,height,width);
	printf("conv1 output dim %d %d %d %d\n",contextV[0]->m_batchSize , conv1V[0].out_channels, conv1V[0].out_height, conv1V[0].out_width);
	printf("pool1 output dim %d %d %d %d\n",contextV[0]->m_batchSize , conv1V[0].out_channels , (conv1V[0].out_height / pool1V[0].stride) , (conv1V[0].out_width / pool1V[0].stride));
	printf("conv2 outut dim %d %d %d %d\n",contextV[0]->m_batchSize, conv2V[0].out_channels , conv2V[0].out_height  , conv2V[0].out_width);
	printf("pool2 output dim %d %d %d %d\n",contextV[0]->m_batchSize , conv2V[0].out_channels , (conv2V[0].out_height / pool2V[0].stride) , (conv2V[0].out_width / pool2V[0].stride));
	printf("fc1 out dim  %d %d\n",contextV[0]->m_batchSize , fc1V[0]->outputs);
	printf("fc1relu out dim  %d %d\n",contextV[0]->m_batchSize , fc1V[0]->outputs);
	printf("fc2 out dim  %d %d\n",contextV[0]->m_batchSize , fc2V[0]->outputs);
	printf("fc2relu out dim  %d %d\n",contextV[0]->m_batchSize , fc2V[0]->outputs);

    // Use SGD to train the network
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter <iterations; ++iter)
    {
        // Train
        int imageid = iter % (train_size / contextV[0]->m_batchSize);

        // Prepare current batch on device
//ssy : I dont need bp, so no need of this
//        checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * pcontext->m_batchSize * width*height*channels],
//                                        sizeof(float) * pcontext->m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
//        checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * pcontext->m_batchSize],
//                                        sizeof(float) * pcontext->m_batchSize, cudaMemcpyHostToDevice));
        
        // Forward propagation
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
        contextV[gpuid]->ForwardPropagation1(
		d_dataV[gpuid], 
		d_conv1V[gpuid], 
		d_pool1V[gpuid], 
		d_conv2V[gpuid], 
		d_pool2V[gpuid], 
		d_fc1V[gpuid], 
		d_fc1reluV[gpuid], 
		d_fc2V[gpuid], 
		d_fc2smaxV[gpuid], 
		d_pconv1V[gpuid], 
		d_pconv1biasV[gpuid], 
		d_pconv2V[gpuid], 
		d_pconv2biasV[gpuid], 
		d_pfc1V[gpuid], 
		d_pfc1biasV[gpuid], 
		d_pfc2V[gpuid], 
		d_pfc2biasV[gpuid],
		d_cudnn_workspaceV[gpuid], 
		d_onevecV[gpuid]
		);
	}
   if(copy) {
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
		//sync n+1 to n
		checkCudaErrors(cudaSetDevice(gpuid));
		size_t sz=sizeof(float) * contextV[gpuid]->m_batchSize * conv1V[gpuid].out_channels * (conv1V[gpuid].out_height / pool1V[gpuid].stride) * (conv1V[gpuid].out_width / pool1V[gpuid].stride);
		printf("coping sz %d\n",sz);
		if(gpuid<num_gpus-1)
	            checkCudaErrors(cudaMemcpyAsync(d_pool1V[gpuid], d_pool1V[gpuid+1], int(fract*sz/2), cudaMemcpyDefault));
		if(gpuid>0)
	            checkCudaErrors(cudaMemcpyAsync(d_pool1V[gpuid]+sz/2, d_pool1V[gpuid-1], int(fract*sz/2), cudaMemcpyDefault));
	}
	checkCudaErrors(cudaDeviceSynchronize());
   }
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
        contextV[gpuid]->ForwardPropagation2(
		d_dataV[gpuid], 
		d_conv1V[gpuid], 
		d_pool1V[gpuid], 
		d_conv2V[gpuid], 
		d_pool2V[gpuid], 
		d_fc1V[gpuid], 
		d_fc1reluV[gpuid], 
		d_fc2V[gpuid], 
		d_fc2smaxV[gpuid], 
		d_pconv1V[gpuid], 
		d_pconv1biasV[gpuid], 
		d_pconv2V[gpuid], 
		d_pconv2biasV[gpuid], 
		d_pfc1V[gpuid], 
		d_pfc1biasV[gpuid], 
		d_pfc2V[gpuid], 
		d_pfc2biasV[gpuid],
		d_cudnn_workspaceV[gpuid], 
		d_onevecV[gpuid]
		);
	}
   if(copy) {
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
		//sync n+1 to n
		checkCudaErrors(cudaSetDevice(gpuid));
		size_t sz=sizeof(float) * contextV[gpuid]->m_batchSize * conv1V[gpuid].out_channels * (conv1V[gpuid].out_height / pool2V[gpuid].stride) * (conv1V[gpuid].out_width / pool2V[gpuid].stride);
		printf("coping sz %d\n",sz);
		if(gpuid<num_gpus-1)
	            checkCudaErrors(cudaMemcpyAsync(d_pool2V[gpuid], d_pool2V[gpuid+1],int(fract*sz/2), cudaMemcpyDefault));
		if(gpuid>0)
	            checkCudaErrors(cudaMemcpyAsync(d_pool2V[gpuid]+sz/2, d_pool2V[gpuid-1], int(fract*sz/2), cudaMemcpyDefault));
	}
	checkCudaErrors(cudaDeviceSynchronize());
   }
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
        contextV[gpuid]->ForwardPropagation3(
		d_dataV[gpuid], 
		d_conv1V[gpuid], 
		d_pool1V[gpuid], 
		d_conv2V[gpuid], 
		d_pool2V[gpuid], 
		d_fc1V[gpuid], 
		d_fc1reluV[gpuid], 
		d_fc2V[gpuid], 
		d_fc2smaxV[gpuid], 
		d_pconv1V[gpuid], 
		d_pconv1biasV[gpuid], 
		d_pconv2V[gpuid], 
		d_pconv2biasV[gpuid], 
		d_pfc1V[gpuid], 
		d_pfc1biasV[gpuid], 
		d_pfc2V[gpuid], 
		d_pfc2biasV[gpuid],
		d_cudnn_workspaceV[gpuid], 
		d_onevecV[gpuid]
		);
	}
        // Backward propagation
// ssy remove this temproally
//        pcontext->Backpropagation(conv1, pool1, conv2, pool2,
//                                d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
//                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
//                                d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias, 
//                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
//
//        // Compute learning rate
//        float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));
//
//        // Update weights
//        pcontext->UpdateWeights(learningRate, conv1, conv2,
//                              d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
//                              d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: width %d fract %f time %f ms\n",width,copy?fract:0.0, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations);
//ssy all save and test    
//    if (FLAGS_save_data)
//    {
//        // Copy trained weights from GPU to CPU
//        checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&pfc1->pneurons[0], d_pfc1, sizeof(float) * pfc1->pneurons.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&pfc1->pbias[0], d_pfc1bias, sizeof(float) * pfc1->pbias.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&pfc2->pneurons[0], d_pfc2, sizeof(float) * pfc2->pneurons.size(), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&pfc2->pbias[0], d_pfc2bias, sizeof(float) * pfc2->pbias.size(), cudaMemcpyDeviceToHost));
//      
//        // Now save data
//        printf("Saving data to file\n");
//        conv1.ToFile("conv1");
//        conv2.ToFile("conv2");
//        pfc1->ToFile("ip1");
//        pfc2->ToFile("ip2");
//    }
//    
//
//    float classification_error = 1.0f;
//
//    int classifications = FLAGS_classify;
//    if (classifications < 0)
//        classifications = (int)test_size;
//    
//    // Test the resulting neural network's classification
//    if (classifications > 0)
//    {
//        // Initialize a TrainingContext structure for testing (different batch size)
//        TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);
//
//        // Ensure correct workspaceSize is allocated for testing
//        if (pcontext->m_workspaceSize < test_context.m_workspaceSize)
//        {
//            checkCudaErrors(cudaFree(d_cudnn_workspace));
//            checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
//        }
//
//        int num_errors = 0;
//        for (int i = 0; i < classifications; ++i)
//        {
//            std::vector<float> data(width * height);
//            // Normalize image to be in [0,1]
//            for (int j = 0; j < width * height; ++j)
//                data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;
//
//            checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
//            
//            // Forward propagate test image
//            test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
//                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
//                                            d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);
//
//            // Perform classification
//            std::vector<float> class_vec(10);
//
//            // Copy back result
//            checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));
//
//            // Determine classification according to maximal response
//            int chosen = 0;
//            for (int id = 1; id < 10; ++id)
//            {
//                if (class_vec[chosen] < class_vec[id]) chosen = id;
//            }
//
//            if (chosen != test_labels[i])
//                ++num_errors;
//        }
//        classification_error = (float)num_errors / (float)classifications;
//
//        printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
//    }
        
    // Free data structures
	for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
    checkCudaErrors(cudaFree(d_dataV[gpuid]));
    checkCudaErrors(cudaFree(d_conv1V[gpuid]));
    checkCudaErrors(cudaFree(d_pool1V[gpuid]));
    checkCudaErrors(cudaFree(d_conv2V[gpuid]));
    checkCudaErrors(cudaFree(d_pool2V[gpuid]));
    checkCudaErrors(cudaFree(d_fc1V[gpuid]));
    checkCudaErrors(cudaFree(d_fc2V[gpuid]));
    checkCudaErrors(cudaFree(d_pconv1V[gpuid]));
    checkCudaErrors(cudaFree(d_pconv1biasV[gpuid]));
    checkCudaErrors(cudaFree(d_pconv2V[gpuid]));
    checkCudaErrors(cudaFree(d_pconv2biasV[gpuid]));
    checkCudaErrors(cudaFree(d_pfc1V[gpuid]));
    checkCudaErrors(cudaFree(d_pfc1biasV[gpuid]));
    checkCudaErrors(cudaFree(d_pfc2V[gpuid]));
    checkCudaErrors(cudaFree(d_pfc2biasV[gpuid]));
    checkCudaErrors(cudaFree(d_gconv1V[gpuid]));
    checkCudaErrors(cudaFree(d_gconv1biasV[gpuid]));
    checkCudaErrors(cudaFree(d_gconv2V[gpuid]));
    checkCudaErrors(cudaFree(d_gconv2biasV[gpuid]));
    checkCudaErrors(cudaFree(d_gfc1V[gpuid]));
    checkCudaErrors(cudaFree(d_gfc1biasV[gpuid]));
//    checkCudaErrors(cudaFree(d_dfc1V[gpuid]));
//    checkCudaErrors(cudaFree(d_dfc1reluV[gpuid]));
    checkCudaErrors(cudaFree(d_gfc2V[gpuid]));
    checkCudaErrors(cudaFree(d_gfc2biasV[gpuid]));
//    checkCudaErrors(cudaFree(d_dfc2V[gpuid]));
//    checkCudaErrors(cudaFree(d_dfc2smaxV[gpuid]));
//    checkCudaErrors(cudaFree(d_dpool1V[gpuid]));
//    checkCudaErrors(cudaFree(d_dconv2V[gpuid]));
//    checkCudaErrors(cudaFree(d_dpool2V[gpuid]));    
    checkCudaErrors(cudaFree(d_labelsV[gpuid]));
//    checkCudaErrors(cudaFree(d_dlossdataV[gpuid]));
    checkCudaErrors(cudaFree(d_onevecV[gpuid]));
    if (d_cudnn_workspaceV[gpuid] != nullptr)
        checkCudaErrors(cudaFree(d_cudnn_workspaceV[gpuid]));
	}
    return 0;
}
