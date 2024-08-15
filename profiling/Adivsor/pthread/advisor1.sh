#!/bin/bash

read -p "Enter the name of the .c or .cpp file to compile: " code_file

if [ -f "$code_file" ]; then
  read -p "Enter the name of the output file: " output_name

  
  if [[ "$code_file" == *.c ]]; then
    gcc -o "$output_name" "$code_file" -lpthread -lm 
  elif [[ "$code_file" == *.cpp ]]; then
    g++ -o "$output_name" "$code_file"
  else
    echo "Invalid file type. Only .c and .cpp files are supported."
    exit 1
  fi

else
  echo "File not found. Please try again."
  exit 1
fi


echo ""
echo "Available Module Advisor : "
module avail | grep advisor

echo ""
echo "Choose an option:"
echo "1. Load latest Advisor"
echo "2. Load a specific Advisor version"

echo ""
read -p "Enter your choice: " advisor_opt

case $advisor_opt in
  1) module load advisor/latest ;;
  2) 
    read -p "Enter a specific Advisor version :" offload_advisor_version
    module load $offload_advisor_version ;;
  *) 
    echo "Invalid option. Exiting."
    exit 1 ;;
esac

my_array=( "1. xehpg_256xve : Intel® Arc™ graphics with 256 vector engines", 
           "2. xehpg_512xve : Intel® Arc™ graphics with 512 vector engines",
           "3.    gen12_tgl : Intel® Iris® Xe graphics",
           "4.    gen12_dg1 : Intel® Iris® Xe MAX graphics",
           "5.    gen11_icl : Intel® Iris® Plus graphics",
           "6.     gen9_gt2 : Intel® HD Graphics 530",
           "7.     gen9_gt3 : Intel® Iris® Graphics 550",
           "8.     gen9_gt4 : Intel® Iris® Pro Graphics 580",
           "9.     Enter Manually.")

my_target_device_list=("xehpg_256xve", "xehpg_512xve","gen12_tgl","gen12_dg1","gen11_icl","gen9_gt2","gen9_gt3","gen9_gt4")

echo ""
echo "================================================"
echo "Available Target device list."
echo "Enter target device name from list only."
echo ""

for element in "${my_array[@]}"; do
  echo "$element"
done

echo "=============================================="

read -p "Enter the target device  name from above :" TARGET_DEVICE

until [[ " ${my_target_device_list[@]}" =~ "$TARGET_DEVICE" ]]
do
  echo ""
  echo "XXXXXXXXXXXX"
  echo "$TARGET_DEVICE is not available in above list!"
  echo "====================================="
  
  echo ""
  read -p "Enter the target device name from above: " TARGET_DEVICE
   
done

echo "Yout Target device :$TARGET_DEVICE is available !"


echo ""
echo "Enter the path for a executable application/file for profiling:"
echo "1. Do you want to enter full path"
echo "2. Enter file name from current dir"

read -p "Enter Here : " APP_EXECUTABLE


if [ ! -f "$APP_EXECUTABLE" ]; then
  echo "Executable file not found. Please try again."
  exit 1
fi


echo ""
echo "we will create a folder in current dir and store report in it "
read -p "Enter a folder name : " OUTPUT_FOLDER_NAME


if [ -d "$OUTPUT_FOLDER_NAME" ]; then
  echo "Folder already exists. Please try again."
  exit 1
fi


echo ""
echo "Executing Advisor command as : "
echo "advisor --collect=offload --accuracy=low --config=$TARGET_DEVICE --project-dir=./$OUTPUT_FOLDER_NAME -- ./$APP_EXECUTABLE"
echo ""
echo "-"
echo "-"
echo "-"

advisor --collect=offload --accuracy=low --config=$TARGET_DEVICE --project-dir=./$OUTPUT_FOLDER_NAME -- ./$APP_EXECUTABLE

xdg-open "./$OUTPUT_FOLDER_NAME/e000/report/advisor-report.html"