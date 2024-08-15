    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <time.h>
    #include <CL/cl.h>

    #define NUM_PATHS 1000000  // Number of paths
    #define NUM_STEPS 1000     // Number of time steps per path

    const float S0 = 100.0f;
    const float K = 100.0f;
    const float B = 120.0f;
    const float T = 1.0f;
    const float r = 0.05f;
    const float sigma = 0.2f;

    const char* kernelSource = R"(
    __kernel void monteCarloSimulation(__global float* results, float S0, float K, float B, float T, float r, float sigma, float dt, float mu, int numSteps) {
        int i = get_global_id(0);
        float S = S0;
        bool breached = false;
        float u1, u2, z;
        int j;

        // Initialize random number generator (this is just a seed for demonstration purposes)
        uint seed = i;
        srand(seed);

        for (j = 0; j < numSteps; ++j) {
            u1 = (float)rand() / RAND_MAX;
            u2 = (float)rand() / RAND_MAX;
            z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

            S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);
            if (S > B) {
                breached = true;
                break;
            }
        }

        float payoff = breached ? 0.0f : fmax(S - K, 0.0f);
        results[i] = payoff * exp(-r * T);
    }
    )";

    char* loadKernelSource(const char* filename) {
        FILE *fp;
        char *source_str;
        size_t source_size;

        fp = fopen(filename, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char*)malloc(0x100000);
        source_size = fread(source_str, 1, 0x100000, fp);
        fclose(fp);

        source_str[source_size] = '\0'; // Ensure null termination
        return source_str;
    }

    int main() {
        float dt = T / NUM_STEPS;
        float mu = r - 0.5f * sigma * sigma;

        cl_platform_id platform_id = NULL;
        cl_device_id device_id = NULL;
        cl_context context = NULL;
        cl_command_queue queue = NULL;
        cl_program program = NULL;
        cl_kernel kernel = NULL;
        cl_mem results_mem = NULL;

        size_t global_work_size = NUM_PATHS;
        size_t local_work_size = 256;

        cl_int ret;

        // Load the kernel source code into the array source_str
        // You can use the embedded kernelSource here instead of loading from a file
        // kernelSource = loadKernelSource("monteCarlo.cl");

        // Get platform and device information
        ret = clGetPlatformIDs(1, &platform_id, NULL);
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

        // Create an OpenCL context
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

        // Create a command queue
        queue = clCreateCommandQueue(context, device_id, 0, &ret);

        // Create a program from the kernel source
        program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        // Create the OpenCL kernel
        kernel = clCreateKernel(program, "monteCarloSimulation", &ret);

        // Allocate space for the results on the host
        float *results = (float*)malloc(NUM_PATHS * sizeof(float));

        // Allocate memory buffers on the device
        results_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_PATHS * sizeof(float), NULL, &ret);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&results_mem);
        ret = clSetKernelArg(kernel, 1, sizeof(float), (void *)&S0);
        ret = clSetKernelArg(kernel, 2, sizeof(float), (void *)&K);
        ret = clSetKernelArg(kernel, 3, sizeof(float), (void *)&B);
        ret = clSetKernelArg(kernel, 4, sizeof(float), (void *)&T);
        ret = clSetKernelArg(kernel, 5, sizeof(float), (void *)&r);
        ret = clSetKernelArg(kernel, 6, sizeof(float), (void *)&sigma);
        ret = clSetKernelArg(kernel, 7, sizeof(float), (void *)&dt);
        ret = clSetKernelArg(kernel, 8, sizeof(float), (void *)&mu);
        ret = clSetKernelArg(kernel, 9, sizeof(int), (void *)&NUM_STEPS);

        // Execute the OpenCL kernel
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

        // Copy the result from the device to the host
        ret = clEnqueueReadBuffer(queue, results_mem, CL_TRUE, 0, NUM_PATHS * sizeof(float), results, 0, NULL, NULL);

        // Calculate the average option price
        float sum = 0.0f;
        for (int i = 0; i < NUM_PATHS; ++i) {
            sum += results[i];
        }
        float monteCarloPrice = sum / NUM_PATHS;

        printf("Estimated Barrier Option Price (Monte Carlo): %f\n", monteCarloPrice);

        // Clean up
        ret = clFlush(queue);
        ret = clFinish(queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(results_mem);
        ret = clReleaseCommandQueue(queue);
        ret = clReleaseContext(context);

        free(results);
        // free(kernelSource); // Not needed here since kernelSource is not used

        return 0;
    }
