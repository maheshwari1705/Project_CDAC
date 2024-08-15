#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>

#define NUM_PATHS 100000  // Number of Monte Carlo paths
#define NUM_STEPS 1000     // Number of time steps per path
#define THREADS 32         // Number of threads

// Parameters for the option and asset
const float S0 = 100.0f; // Initial asset price
const float K = 100.0f;  // Strike price
const float B = 120.0f;  // Barrier level
const float T = 1.0f;    // Time to maturity
const float r = 0.05f;   // Risk-free rate
const float sigma = 0.2f; // Volatility

// Structure to pass arguments to threads
typedef struct {
    float* results;
    int start;
    int end;
    float dt;
    float mu;
    float sigma;
    float barrier;
    float strike;
    float r;
    float T;
} ThreadData;

// Function to generate random numbers using Box-Muller transform
float boxMuller(float u1, float u2) {
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// Monte Carlo simulation function for each thread
void* monteCarloThread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    float* results = data->results;
    int start = data->start;
    int end = data->end;
    float dt = data->dt;
    float mu = data->mu;
    float sigma = data->sigma;
    float barrier = data->barrier;
    float strike = data->strike;
    float r = data->r;
    float T = data->T;

    for (int i = start; i < end; ++i) {
        float S = S0;
        bool breached = false;

        for (int j = 0; j < NUM_STEPS; ++j) {
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

    pthread_exit(NULL);
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

    // Create and initialize pthreads
    pthread_t threads[THREADS];
    ThreadData threadData[THREADS];
    int pathsPerThread = (NUM_PATHS + THREADS - 1) / THREADS;

    // Start timing
    clock_t start = clock();

    for (int i = 0; i < THREADS; ++i) {
        threadData[i].results = results;
        threadData[i].start = i * pathsPerThread;
        threadData[i].end = (i + 1) * pathsPerThread > NUM_PATHS ? NUM_PATHS : (i + 1) * pathsPerThread;
        threadData[i].dt = dt;
        threadData[i].mu = mu;
        threadData[i].sigma = sigma;
        threadData[i].barrier = B;
        threadData[i].strike = K;
        threadData[i].r = r;
        threadData[i].T = T;

        pthread_create(&threads[i], NULL, monteCarloThread, (void*)&threadData[i]);
    }

    // Join threads
    for (int i = 0; i < THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    // End timing
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
