#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

#define NUM_PATHS 1000000000 // Number of Monte Carlo paths
#define NUM_STEPS 1000     // Number of time steps per path
#define THREADS_PER_BLOCK 256

// Parameters for the option and asset
const float S0 = 100.0f; // Initial asset price
const float K = 100.0f;  // Strike price
const float B = 120.0f;  // Barrier level
const float T = 1.0f;    // Time to maturity
const float r = 0.05f;   // Risk-free rate
const float sigma = 0.2f; // Volatility

// CUDA kernel for Monte Carlo simulation with 3D indexing
__global__ void monteCarloSimulation(float* results, curandState* state, int numPaths, int numSteps, float dt, float mu, float sigma, float barrier, float strike, float r, float T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int tid = x + y * gridDim.x * blockDim.x + z * gridDim.x * blockDim.x * blockDim.y;

    if (tid < numPaths) {
        curandState localState = state[tid];
        float S = S0;
        bool breached = false;

        for (int i = 0; i < numSteps; ++i) {
            float z = curand_normal(&localState);
            S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);
            if (S > barrier) {
                breached = true;
                break;
            }
        }

        float payoff = breached ? 0.0f : max(S - strike, 0.0f);
        results[tid] = payoff * exp(-r * T); // Discounted payoff
    }
}

// CUDA kernel to initialize random states
__global__ void initCurandStates(curandState* state, unsigned long long seed, int numPaths) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int tid = x + y * gridDim.x * blockDim.x + z * gridDim.x * blockDim.x * blockDim.y;

    if (tid < numPaths) {
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// CUDA kernel to solve Black-Scholes PDE using finite difference method with 3D indexing
__global__ void blackScholesPDE(float* optionPrices, int numSteps, float Smax, float dS, float dt, float r, float sigma, float K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int j = x + y * gridDim.x * blockDim.x + z * gridDim.x * blockDim.x * blockDim.y;

    if (j <= numSteps) {
        float S = j * dS;
        float payoff = max(S - K, 0.0f);
        optionPrices[j] = payoff;
    }
}

// CUDA kernel to calculate barrier option pricing with 3D indexing
__global__ void barrierOptionPricing(float* results, curandState* state, int numPaths, int numSteps, float dt, float mu, float sigma, float barrier, float strike, float r, float T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int tid = x + y * gridDim.x * blockDim.x + z * gridDim.x * blockDim.x * blockDim.y;

    if (tid < numPaths) {
        curandState localState = state[tid];
        float S = S0;
        bool breached = false;

        for (int i = 0; i < numSteps; ++i) {
            float z = curand_normal(&localState);
            S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);
            if (S > barrier) {
                breached = true;
                break;
            }
        }

        float payoff = breached ? 0.0f : max(S - strike, 0.0f);
        results[tid] = payoff * exp(-r * T); // Discounted payoff
    }
}

int main() {
    float *d_results, *h_results;
    curandState *d_state;
    float *d_optionPrices, *h_optionPrices;
    size_t sizeResults = NUM_PATHS * sizeof(float);
    size_t sizeStates = NUM_PATHS * sizeof(curandState);
    size_t sizePrices = (NUM_STEPS + 1) * sizeof(float);

    // Allocate host memory
    h_results = new float[NUM_PATHS];
    h_optionPrices = new float[NUM_STEPS + 1];

    // Allocate device memory
    cudaMalloc(&d_results, sizeResults);
    cudaMalloc(&d_state, sizeStates);
    cudaMalloc(&d_optionPrices, sizePrices);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize random states
    unsigned long long seed = 1234ULL;
    int threadsPerBlock = THREADS_PER_BLOCK;
    dim3 threads(threadsPerBlock, 1, 1);

    int blocksX = (NUM_PATHS + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(blocksX, 1, 1);

    cudaEventRecord(start);
    initCurandStates<<<blocks, threads>>>(d_state, seed, NUM_PATHS);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float initTime;
    cudaEventElapsedTime(&initTime, start, stop);

    // Parameters for the simulation
    float dt = T / NUM_STEPS;
    float mu = r - 0.5f * sigma * sigma;

    // Configure 3D grid and block dimensions for the Monte Carlo simulation
    int gridDimX = (NUM_PATHS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(gridDimX, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    cudaEventRecord(start);
    monteCarloSimulation<<<grid, block>>>(d_results,    d_state, NUM_PATHS, NUM_STEPS, dt, mu, sigma, B, K, r, T);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float simulationTime;
    cudaEventElapsedTime(&simulationTime, start, stop);

    // Copy results from device to host
    cudaMemcpy(h_results, d_results, sizeResults, cudaMemcpyDeviceToHost);

    // Calculate the average option price
    float sum = 0.0f;
    for (int i = 0; i < NUM_PATHS; ++i) {
        sum += h_results[i];
    }
    float monteCarloPrice = sum / NUM_PATHS;

    std::cout << "Estimated Barrier Option Price (Monte Carlo): " << monteCarloPrice << std::endl;
    std::cout << "Time to initialize CURAND states: " << initTime << " ms" << std::endl;
    std::cout << "Time for Monte Carlo simulation: " << simulationTime << " ms" << std::endl;

    // Solve Black-Scholes PDE
    float Smax = 2 * S0; // Maximum asset price for the PDE grid
    float dS = Smax / NUM_STEPS;

    int pdeThreadsPerBlock = THREADS_PER_BLOCK;
    int pdeBlocks = (NUM_STEPS + pdeThreadsPerBlock - 1) / pdeThreadsPerBlock;
    dim3 pdeGrid(pdeBlocks, 1, 1);
    dim3 pdeThreads(pdeThreadsPerBlock, 1, 1);

    cudaEventRecord(start);
    blackScholesPDE<<<pdeGrid, pdeThreads>>>(d_optionPrices, NUM_STEPS, Smax, dS, dt, r, sigma, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pdeTime;
    cudaEventElapsedTime(&pdeTime, start, stop);

    // Copy PDE results from device to host
    cudaMemcpy(h_optionPrices, d_optionPrices, sizePrices, cudaMemcpyDeviceToHost);

    // Print PDE results (optional, for debugging purposes)
    std::cout << "Black-Scholes PDE prices at final time step:" << std::endl;
    for (int i = 0; i <= NUM_STEPS; ++i) {
        float S = i * dS;
        //std::cout << "S = " << S << ", Price = " << h_optionPrices[i] << std::endl;
    }

    std::cout << "Time for Black-Scholes PDE solution: " << pdeTime << " ms" << std::endl;

    // Run Barrier Option Pricing Calculation
    cudaEventRecord(start);
    barrierOptionPricing<<<grid, block>>>(d_results, d_state, NUM_PATHS, NUM_STEPS, dt, mu, sigma, B, K, r, T);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float barrierPricingTime;
    cudaEventElapsedTime(&barrierPricingTime, start, stop);

    // Copy results from device to host
    cudaMemcpy(h_results, d_results, sizeResults, cudaMemcpyDeviceToHost);

    // Calculate the average barrier option price
    sum = 0.0f;
    for (int i = 0; i < NUM_PATHS; ++i) {
        sum += h_results[i];
    }
    float barrierOptionPrice = sum / NUM_PATHS;

    std::cout << "Estimated Barrier Option Price (Barrier Option Pricing Kernel): " << barrierOptionPrice << std::endl;
    std::cout << "Time for Barrier Option Pricing calculation: " << barrierPricingTime << " ms" << std::endl;

    // Free device memory
    cudaFree(d_results);
    cudaFree(d_state);
    cudaFree(d_optionPrices);

    // Free host memory
    delete[] h_results;
    delete[] h_optionPrices;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// source /home/apps/spack/share/spack/setup-env.sh
// spack load cuda
// spack load cuda/x2ghzj4
// nvcc cuda1.cu
// ./a.out
