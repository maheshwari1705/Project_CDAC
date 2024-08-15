#!/bin/bash

# Ask the user for the source file name
echo "Enter the source file name (e.g., my-application.c): "
read source_file

# Compile the program with profiling enabled
echo "Compiling $source_file with profiling enabled..."
gcc -pg -o my-application $source_file -lpthread -lm

# Run the program to generate profiling data
echo "Running my-application to generate profiling data..."
./my-application

# Run gprof to analyze the profiling data
echo "Running gprof to analyze the profiling data..."
gprof my-application gmon.out > analysis.txt

# Open the analysis file in vi
echo "Opening analysis.txt in vi..."
cat analysis.txt