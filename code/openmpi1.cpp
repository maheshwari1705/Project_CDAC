#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// Parameters for the option and asset
const float S0 = 100.0f; // Initial asset price
const float K = 100.0f;  // Strike price
const float B = 120.0f;  // Barrier level
const float T = 1.0f;    // Time to maturity
const float r = 0.05f;   // Risk-free rate
const float sigma = 0.2f; // Volatility

// Function to simulate one path for Monte Carlo
float simulatePath(int numSteps, float dt, float mu, float sigma, float barrier, float strike, float r, float T) {
    float S = S0;
    bool breached = false;
    
    for (int i = 0; i < numSteps; ++i) {
        float z = ((float) rand() / RAND_MAX) * 2.0f - 1.0f; // Uniform random number
        S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);
        if (S > barrier) {
            breached = true;
            break;
        }
    }
    
    float payoff = breached ? 0.0f : std::max(S - strike, 0.0f);
    return payoff * exp(-r * T); // Discounted payoff
}

// Function to solve Black-Scholes PDE (simplified example)
std::vector<float> blackScholesPDE(int numSteps, float Smax, float dS, float dt, float r, float sigma, float K) {
    std::vector<float> optionPrices(numSteps + 1);
    
    for (int i = 0; i <= numSteps; ++i) {
        float S = i * dS;
        float payoff = std::max(S - K, 0.0f);
        optionPrices[i] = payoff;
    }
    
    return optionPrices;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int NUM_PATHS = 10000000;
    const int NUM_STEPS = 1000;
    
    float dt = T / NUM_STEPS;
    float mu = r - 0.5f * sigma * sigma;
    
    int localNumPaths = NUM_PATHS / size;
    int startPath = rank * localNumPaths;
    
    // Allocate memory for results
    std::vector<float> localResults(localNumPaths);
    
    // Timing
    double startTime = MPI_Wtime();

    // Monte Carlo Simulation
    for (int i = 0; i < localNumPaths; ++i) {
        localResults[i] = simulatePath(NUM_STEPS, dt, mu, sigma, B, K, r, T);
    }
    
    double endTime = MPI_Wtime();
    double computationTime = endTime - startTime;
    
    // Gather results from all processes
    std::vector<float> allResults(NUM_PATHS);
    MPI_Gather(localResults.data(), localNumPaths, MPI_FLOAT, allResults.data(), localNumPaths, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Compute average price
        float sum = std::accumulate(allResults.begin(), allResults.end(), 0.0f);
        float monteCarloPrice = sum / NUM_PATHS;

        std::cout << "Estimated Barrier Option Price (Monte Carlo): " << monteCarloPrice << std::endl;
        std::cout << "Time for Monte Carlo simulation: " << computationTime << " seconds" << std::endl;

        // Solve Black-Scholes PDE
        float Smax = 2 * S0; // Maximum asset price for the PDE grid
        float dS = Smax / NUM_STEPS;

        startTime = MPI_Wtime();
        auto optionPrices = blackScholesPDE(NUM_STEPS, Smax, dS, dt, r, sigma, K);
        endTime = MPI_Wtime();
        double pdeTime = endTime - startTime;

        std::cout << "Black-Scholes PDE prices at final time step:" << std::endl;
        for (int i = 0; i <= NUM_STEPS; ++i) {
            float S = i * dS;
            //std::cout << "S = " << S << ", Price = " << optionPrices[i] << std::endl;
        }
        std::cout << "Time for Black-Scholes PDE solution: " << pdeTime << " seconds" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
// source /home/apps/spack/share/spack/setup-env.sh
// spack load openmpi
// spack load openmpi/inq7hh5
// mpic++ openmpi1.cpp
// mpirun -np 10 ./a.out
