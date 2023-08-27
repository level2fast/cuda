
#include <cuda.h>
#include <cublas_v2.h>

const static float dt = 1.0E-01f;

// Utility CUDA kernel functions
__device__ float activation_function(float x)
{
    return 1 / (1 + exp(-x));
}

__global__ void apply_activation_function(float *input, float *output, const int N)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        output[idx] = activation_function(input[idx]);
    }
}

//Function: makeError
//params:
// err    - the preactivation value used by backpropagation function
// output - final neuron at the end fully connected layer
// Y      - the label for the current image 
// N      - Number of outputs in our fully connected layer
__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        output[idx] += dt * grad[idx];
    }
}




// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 5*5*6*24*24;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 5);
        const int i2 = ((idx /= 5    ) % 5);
        const int i3 = ((idx /= 5    ) % 6);
        const int i4 = ((idx /= 6    ) % 24);
        const int i5 = ((idx /= 24   ) % 24);

        atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
    }
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 24);
        const int i3 = ((idx /= 24   ) % 24);

        preact[i1][i2][i3] += bias[i1];
    }
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][8][8], float weight[1][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 3*3*6*8*8;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 3);
        const int i2 = ((idx /= 3    ) % 3);
        const int i3 = ((idx /= 3    ) % 6);
        const int i4 = ((idx /= 6    ) % 8);
        const int i5 = ((idx /= 8   ) % 8);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 3 + i1][i5 * 3 + i2]);
    }
}

__global__ void fp_bias_s1(float preact[6][8][8], float bias[1])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*8*8;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 8);
        const int i3 = ((idx /= 8    ) % 8);

        preact[i1][i2][i3] += bias[0];
    }
}

__global__ void fp_preact_f(float input[6][3][3], float preact[10], float weight[10][6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 10*6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1     ) % 10);
        const int i2 = ((idx /= 10    ) % 6);
        const int i3 = ((idx /= 6     ) % 3);
        const int i4 = ((idx /= 3     ) % 3);

        atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
    }
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 10;

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        preact[idx] += bias[idx];
    }
}




// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][6][3][3], float d_preact[10], float p_output[6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 10*6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1     ) % 10);
        const int i2 = ((idx /= 10    ) % 6);
        const int i3 = ((idx /= 6     ) % 3);
        const int i4 = ((idx /= 3     ) % 3);

        d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
    }
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 10;

    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
        bias[idx] += dt * d_preact[idx];
    }
}

__global__ void bp_output_s1(float d_output[6][8][8], float n_weight[6][3][3], float nd_preact[6][6][6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*3*3*6*6*6;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);
        const int i4 = ((idx /= 3    ) % 6);
        const int i5 = ((idx /= 6    ) % 6);
        const int i6 = ((idx /= 6    ) % 6);

        atomicAdd(&d_output[i4][i5 * 3 + i2][i6 * 3 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void bp_preact_s1(float d_preact[6][8][8], float d_output[6][8][8], float preact[6][8][8])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*8*8;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 8);
        const int i3 = ((idx /= 8    ) % 8);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void bp_weight_s1(float d_weight[1][3][3], float d_preact[6][8][8], float p_output[6][24][24])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 1*3*3*6*6*6;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 1);
        const int i2 = ((idx /= 1    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);
        const int i4 = ((idx /= 3    ) % 6);
        const int i5 = ((idx /= 6    ) % 6);
        const int i6 = ((idx /= 6    ) % 6);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 3 + i2][i6 * 3 + i3]);
    }
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][8][8])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*8*8;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1     ) % 6);
        const int i2 = ((idx /= 6     ) % 8);
        const int i3 = ((idx /= 8    ) % 8);

        atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
    }
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][3][3], float nd_preact[6][8][8])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 1*3*3*6*6*6;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 1);
        const int i2 = ((idx /= 1    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);
        const int i4 = ((idx /= 3    ) % 6);
        const int i5 = ((idx /= 6    ) % 6);
        const int i6 = ((idx /= 6    ) % 6);

        atomicAdd(&d_output[i4][i5 * 3 + i2][i6 * 3 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1     ) % 6);
        const int i2 = ((idx /= 6     ) % 24);
        const int i3 = ((idx /= 24    ) % 24);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*5*5*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 5);
        const int i3 = ((idx /= 5    ) % 5);
        const int i4 = ((idx /= 5    ) % 24);
        const int i5 = ((idx /= 24   ) % 24);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
    }
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 24);
        const int i3 = ((idx /= 24   ) % 24);

        atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
    }
}


///////// POOL LAYER
__global__ void bp_output_s2(float d_output[6][3][3], float n_weight[10][6][3][3], float nd_preact[10])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 10*6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 10);
        const int i2 = ((idx /= 10   ) % 6);
        const int i3 = ((idx /= 6    ) % 3);
        const int i4 = ((idx /= 3    ) % 3);

        atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
    }
}

__global__ void bp_preact_s2(float d_preact[6][3][3], float d_output[6][3][3], float preact[6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void bp_weight_s2(float d_weight[1][2][2], float d_preact[6][3][3], float p_output[6][6][6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 1*2*2*6*3*3;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 1);
        const int i2 = ((idx /= 1    ) % 2);
        const int i3 = ((idx /= 2    ) % 2);
        const int i4 = ((idx /= 2    ) % 6);
        const int i5 = ((idx /= 6    ) % 3);
        const int i6 = ((idx /= 3    ) % 3);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 2 + i2][i6 * 2 + i3]);
    }
}

__global__ void bp_bias_s2(float bias[1], float d_preact[6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*3*3;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);

        atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
    }
}

/////////   CONV LAYER  TODO
__global__ void bp_output_c2(float d_output[6][6][6], float n_weight[6][2][2], float nd_preact[6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*2*2*6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 2);
        const int i3 = ((idx /= 2    ) % 2);
        const int i4 = ((idx /= 2    ) % 6);
        const int i5 = ((idx /= 6    ) % 3);
        const int i6 = ((idx /= 3    ) % 3);

        atomicAdd(&d_output[i4][i5 * 2 + i2][i6 * 2 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void bp_preact_c2(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 6);
        const int i3 = ((idx /= 6    ) % 6);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void bp_weight_c2(float d_weight[6][3][3], float d_preact[6][6][6], float p_output[6][8][8])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*3*3*6*6;
    const float d = pow(6.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);
        const int i4 = ((idx /= 3    ) % 6);
        const int i5 = ((idx /= 6    ) % 6);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i1][i4 + i2][i5 + i3] / d);
    }
}

__global__ void bp_bias_c2(float bias[6], float d_preact[6][6][6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*6*6;
    const float d = pow(6.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 6);
        const int i3 = ((idx /= 6    ) % 6);

        atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
    }
}

// Forward propagation kernels
__global__ void fp_preact_c2(float input[6][8][8], float preact[6][6][6], float weight[6][3][3])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 3*3*6*6*6;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 3);
        const int i2 = ((idx /= 3    ) % 3);
        const int i3 = ((idx /= 3    ) % 6);
        const int i4 = ((idx /= 6    ) % 6);
        const int i5 = ((idx /= 6    ) % 6);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 + i1][i5 + i2]);
    }
}

__global__ void fp_bias_c2(float preact[6][6][6], float bias[6])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 6);
        const int i3 = ((idx /= 6    ) % 6);

        preact[i1][i2][i3] += bias[i1];
    }
}

__global__ void fp_preact_s2(float input[6][6][6], float preact[6][3][3], float weight[1][2][2])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 3*3*6*2*2;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 3);
        const int i2 = ((idx /= 3    ) % 3);
        const int i3 = ((idx /= 3    ) % 6);
        const int i4 = ((idx /= 6    ) % 2);
        const int i5 = ((idx /= 2    ) % 2);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2]);
    }
}

__global__ void fp_bias_s2(float preact[6][3][3], float bias[1])
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;

    const int N = 6*3*3;

    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1    ) % 6);
        const int i2 = ((idx /= 6    ) % 3);
        const int i3 = ((idx /= 3    ) % 3);

        preact[i1][i2][i3] += bias[0];
    }
}
