#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>  

#define NUM_PATHS 10000000  // Reduced number of paths for computational feasibility
#define NUM_STEPS 1000     // Number of time steps per path

// Parameters for the option and asset
const float S0 = 100.0f; // Initial asset price
const float K = 100.0f;  // Strike price
const float B = 120.0f;  // Barrier level
const float T = 1.0f;    // Time to maturity
const float r = 0.05f;   // Risk-free rate
const float sigma = 0.2f; // Volatility

// Function to generate random numbers using Box-Muller transform
float boxMuller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// Monte Carlo simulation function
void monteCarloSimulation(float* results, int numPaths, int numSteps, float dt, float mu, float sigma, float barrier, float strike, float r, float T) {
    for (int i = 0; i < numPaths; ++i) {
        float S = S0;
        bool breached = false;
        
        for (int j = 0; j < numSteps; ++j) {
            float u1 = rand() / (float)RAND_MAX;
            float u2 = rand() / (float)RAND_MAX;
            float z = boxMuller(u1, u2);
            S *= expf((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
            if (S > barrier) {
                breached = true;
                break;
            }
        }
        
        float payoff = breached ? 0.0f : fmaxf(S - strike, 0.0f);
        results[i] = payoff * expf(-r * T); // Discounted payoff
    }
}

// Function to solve Black-Scholes PDE using finite difference method
void blackScholesPDE(float* optionPrices, int numSteps, float Smax, float dS, float dt, float r, float sigma, float K) {
    for (int j = 0; j <= numSteps; ++j) {
        float S = j * dS;
        float payoff = fmaxf(S - K, 0.0f);
        optionPrices[j] = payoff;
    }
}

// Main function
int main() {
    float *results, *optionPrices;
    size_t sizeResults = NUM_PATHS * sizeof(float);
    size_t sizePrices = (NUM_STEPS + 1) * sizeof(float);

    // Allocate memory
    results = (float*)malloc(sizeResults);
    optionPrices = (float*)malloc(sizePrices);

    // Parameters for the simulation
    float dt = T / NUM_STEPS;
    float mu = r - 0.5f * sigma * sigma;

    // Seed the random number generator
    srand(time(NULL));

    // Run Monte Carlo simulation
    clock_t start = clock();
    monteCarloSimulation(results, NUM_PATHS, NUM_STEPS, dt, mu, sigma, B, K, r, T);
    clock_t end = clock();
    double simulationTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Calculate the average option price
    float sum = 0.0f;
    for (int i = 0; i < NUM_PATHS; ++i) {
        sum += results[i];
    }
    float monteCarloPrice = sum / NUM_PATHS;

    printf("Estimated Barrier Option Price (Monte Carlo): %f\n", monteCarloPrice);
    printf("Time for Monte Carlo simulation: %f seconds\n", simulationTime);

    // Solve Black-Scholes PDE
    float Smax = 2 * S0; // Maximum asset price for the PDE grid
    float dS = Smax / NUM_STEPS;

    start = clock();
    blackScholesPDE(optionPrices, NUM_STEPS, Smax, dS, dt, r, sigma, K);
    end = clock();
    double pdeTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Print PDE results (optional, for debugging purposes)
    printf("Black-Scholes PDE prices at final time step:\n");
    for (int i = 0; i <= NUM_STEPS; ++i) {
        float S = i * dS;
        // Uncomment below line to print the values
        //printf("S = %f, Price = %f\n", S, optionPrices[i]);
    }
    printf("Time for Black-Scholes PDE solution: %f seconds\n", pdeTime);

    // Free allocated memory
    free(results);
    free(optionPrices);

    return 0;
}
//gcc cpp1.c
