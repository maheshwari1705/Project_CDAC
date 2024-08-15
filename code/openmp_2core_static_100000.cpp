#include <omp.h>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

#define NUM_PATHS 10000000  // Number of Monte Carlo paths
#define NUM_STEPS 1000      // Number of time steps per path
#define THREADS 2           // Number of OpenMP threads

// Static chunk size
#define STATIC_CHUNK_SIZE 100000

// Parameters for the option and asset
const float S0 = 100.0f;  // Initial asset price
const float K = 100.0f;   // Strike price
const float B = 120.0f;   // Barrier level
const float T = 1.0f;     // Time to maturity
const float r = 0.05f;    // Risk-free rate
const float sigma = 0.2f; // Volatility

// Monte Carlo simulation for barrier options
void monteCarloSimulation(std::vector<float>& results, int numPaths, int numSteps, float dt, float mu, float sigma, float barrier, float strike, float r, float T) {
    #pragma omp parallel num_threads(THREADS)
    {
        // Thread-local random generator and distribution
        std::default_random_engine generator(omp_get_thread_num());
        std::normal_distribution<float> distribution(0.0, 1.0);

        #pragma omp for schedule(static, STATIC_CHUNK_SIZE)
        for (int tid = 0; tid < numPaths; ++tid) {
            float S = S0;
            bool breached = false;

            for (int i = 0; i < numSteps; ++i) {
                float z = distribution(generator);
                S *= expf((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
                if (S > barrier) {
                    breached = true;
                    break;
                }
            }

            float payoff = breached ? 0.0f : fmaxf(S - strike, 0.0f);
            results[tid] = payoff * expf(-r * T); // Discounted payoff
        }
    }
}

// Solve Black-Scholes PDE using finite difference method
void blackScholesPDE(std::vector<float>& optionPrices, int numSteps, float Smax, float dS, float dt, float r, float sigma, float K) {
    std::vector<float> oldPrices(numSteps + 1);
    std::vector<float> newPrices(numSteps + 1);

    // Initialize option prices at maturity
    #pragma omp parallel for schedule(static, STATIC_CHUNK_SIZE)
    for (int j = 0; j <= numSteps; ++j) {
        float S = j * dS;
        float payoff = fmaxf(S - K, 0.0f);
        oldPrices[j] = payoff;
    }

    // Backward induction to solve PDE
    for (int step = numSteps - 1; step >= 0; --step) {
        #pragma omp parallel for schedule(static, STATIC_CHUNK_SIZE)
        for (int j = 1; j < numSteps; ++j) {
            float S = j * dS;
            float price = expf(-r * dt) * (
                (oldPrices[j + 1] * 0.5f * dt * (sigma * sigma * S * S + (r - 0.5f * sigma * sigma) * S) +
                (1 - dt * sigma * sigma * S * S - r * dt) * oldPrices[j] +
                oldPrices[j - 1] * 0.5f * dt * (sigma * sigma * S * S - (r - 0.5f * sigma * sigma) * S)) / (1 + dt * sigma * sigma * S * S));
            newPrices[j] = price;
        }
        std::swap(oldPrices, newPrices);
    }

    #pragma omp parallel for schedule(static, STATIC_CHUNK_SIZE)
    for (int j = 0; j <= numSteps; ++j) {
        optionPrices[j] = oldPrices[j];
    }
}

int main() {
    std::vector<float> results(NUM_PATHS);
    std::vector<float> optionPrices(NUM_STEPS + 1);

    // Timing variables
    double startTime, endTime;

    // Parameters for the simulation
    float dt = T / NUM_STEPS;
    float mu = r - 0.5f * sigma * sigma;

    // Print chunk size for static scheduling
    std::cout << "Static Scheduling Chunk Size: " << STATIC_CHUNK_SIZE << std::endl;

    // Monte Carlo simulation
    startTime = omp_get_wtime();
    monteCarloSimulation(results, NUM_PATHS, NUM_STEPS, dt, mu, sigma, B, K, r, T);
    endTime = omp_get_wtime();
    double simulationTime = endTime - startTime;

    // Calculate the average option price
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < NUM_PATHS; ++i) {
        sum += results[i];
    }
    float monteCarloPrice = sum / NUM_PATHS;

    std::cout << "Estimated Barrier Option Price (Monte Carlo): " << monteCarloPrice << std::endl;
    std::cout << "Time for Monte Carlo simulation: " << simulationTime << " seconds" << std::endl;

    // Solve Black-Scholes PDE
    float Smax = 2 * S0; // Maximum asset price for the PDE grid
    float dS = Smax / NUM_STEPS;

    startTime = omp_get_wtime();
    blackScholesPDE(optionPrices, NUM_STEPS, Smax, dS, dt, r, sigma, K);
    endTime = omp_get_wtime();
    double pdeTime = endTime - startTime;

    // Print PDE results (optional, for debugging purposes)
    std::cout << "Black-Scholes PDE prices at final time step:" << std::endl;
    for (int i = 0; i <= NUM_STEPS; ++i) {
        float S = i * dS;
        //std::cout << "S = " << S << ", Price = " << optionPrices[i] << std::endl;
    }

    std::cout << "Time for Black-Scholes PDE solution: " << pdeTime << " seconds" << std::endl;

    return 0;
}
