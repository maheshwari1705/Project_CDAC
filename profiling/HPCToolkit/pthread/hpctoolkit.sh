#!/bin/bash

# Load the Spack environment
source /home/apps/spack/share/spack/setup-env.sh

# Load the HPCToolkit module using Spack
spack load hpctoolkit

# Print the HPCToolkit version
hpcrun -h

# Load the HPCToolkit module
module load hpctoolkit

echo "Enter a list of events (separated by spaces):"
echo "Available events: REALTIME, CPUTIME, perf::CACHE-MISSES, MEMLEAK, IO"
read -r events

if echo "$events" | grep -q "REALTIME" && echo "$events" | grep -q "CPUTIME"; then
  echo "Error: REALTIME and CPUTIME cannot be input together"
  exit 1
fi

echo "Do you want to print trace view? (y/n)"
read -r trace_view

echo "Enter the absolute path to the executable file:"
read -r executable_path

# Check if the executable file is executable
if [ ! -x "$executable_path" ]; then
  echo "Error: $executable_path is not an executable file"
  exit 1
fi

hpcrun_cmd="hpcrun "
for event in $events; do
  hpcrun_cmd+=" -e $event"
done
hpcrun_cmd+=" -t $executable_path"

echo "$hpcrun_cmd" 

ulimit -s unlimited
$hpcrun_cmd

executable_name=$(basename "$executable_path")

hpcstruct hpctoolkit-$executable_name-measurements/

hpcprof hpctoolkit-$executable_name-measurements/

if [ "$trace_view" = "y" ]; then
  hpcviewer hpctoolkit-$executable_name-database/ &
else
  hpcviewer hpctoolkit-$executable_name-database/
fi