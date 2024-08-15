#!/bin/bash

# Get the list of available advisor modules
available_modules=$(module avail advisor | grep -v "No module" | awk '{print $1}')

# Ask the user to select an advisor module
echo "Select an advisor module:"
select module in $available_modules; do
  break
done

# Load the selected module
module load $module

# Define the allowed GPU names
allowed_gpus=(xehpg_256xve xehpg_512xve gen12_tgl gen12_tg1 gen11_icl gen9_tg2 gen9_tg3 gen9_gt4)

# Ask the user for the config (target GPU name)
while true; do
  echo "Enter the target GPU name (one of the following):"
  echo "${allowed_gpus[@]}"
  read target_gpu

  # Check if the input is in the allowed list
  if [[ " ${allowed_gpus[*]} " =~ " $target_gpu " ]]; then
    break
  else
    echo "Invalid GPU name. Please try again."
  fi
done

# Ask the user for the path to the executable file
echo "Enter the path to the executable file: "
read executable_path

# Ask the user for the path to the project directory
echo "Enter the path to the project directory: "
read project_dir

#!/bin/bash

echo "Compile Code"
echo "-----------"

echo "Enter the file name with extension (.c or .cpp or .cu): "
read file_name

if [ "${file_name##*.}" == "c" ]; then
    echo "Compiling C code..."
    output_name="${file_name%.c}"
    gcc -o "$output_name" "$file_name" -lpthread -lm
    echo "Compilation successful! Output file: $output_name"
elif [ "${file_name##*.}" == "cpp" ]; then
    echo "Compiling C++ code..."
    output_name="${file_name%.cpp}"
    g++ -o "$output_name" "$file_name"
    echo "Compilation successful! Output file: $output_name"
elif [ "${file_name##*.}" == "cu" ]; then
    echo "Compiling cuda code..."
    output_name="${file_name%.cu}"
    nvcc -o "$output_name" "$file_name"
    echo "Compilation successful! Output file: $output_name"

else
    echo "Invalid file extension. Please enter a .c or .cpp file."
fi

# Create the advisor command
advisor_cmd="advisor --collect=offload --config=$target_gpu --project-dir=$project_dir -- $executable_path"

# Run the advisor command and generate the report
echo "Running advisor command: $advisor_cmd"
$advisor_cmd > $project_dir/advisor_report.txt 2>&1

echo "Advisor report generated in $project_dir/advisor_report.txt"

open_report=$(xdg-open advisor-report.html)
$open_report