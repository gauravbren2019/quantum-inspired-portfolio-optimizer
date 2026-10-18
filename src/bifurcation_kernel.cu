#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__ void bsb_step_kernel(
    float* x, float* y, const float* field, int n_assets, int bits, 
    float K, float delta0, float xi0, float p_t, float sigma_t, float dt, unsigned long seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n_osc = n_assets * bits;
    if (k >= n_osc) return;

    int i = k / bits;
    int b = k % bits;
    float bit_weight = powf(2.0f, (float)b);
    float local_field = bit_weight * field[i];

    curandState state;
    curand_init(seed, k, 0, &state);
    float stoch = sigma_t * curand_normal(&state);

    float xi = x[k];
    float nonlinear = -(K * xi * xi - p_t + delta0) * xi;
    float coupling = -xi0 * local_field;

    y[k] += (nonlinear + coupling + stoch) * dt;
    x[k] += y[k] * dt;

    if (x[k] > 1.0f)  { x[k] = 1.0f;  y[k] = 0.0f; }
    if (x[k] < -1.0f) { x[k] = -1.0f; y[k] = 0.0f; }
}

extern "C" void launch_bsb_step(float* x, float* y, const float* field, int n_assets, int bits,
                                 float K, float delta0, float xi0, float p_t, float sigma_t, float dt, unsigned long seed)
{
    int n_osc = n_assets * bits;
    int threads = 256;
    int blocks = (n_osc + threads - 1) / threads;
    bsb_step_kernel<<<blocks, threads>>>(x, y, field, n_assets, bits, K, delta0, xi0, p_t, sigma_t, dt, seed);
}
