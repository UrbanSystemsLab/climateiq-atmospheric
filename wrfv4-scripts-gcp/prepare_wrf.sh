#!/bin/bash

# Function to remove spaces from input string
remove_spaces() {
    echo "${1// /}"
}

# Function to prompt user for file/directory location
prompt_location() {
    local prompt_message=$1
    local location=""
    while [ ! -e "$location" ]; do
        read -p "$prompt_message: " location
    done
    echo "$location"
}

# Function to generate simulation directory name with numeric suffix
generate_sim_dir_name() {
    local base_dir="$1"
    local base_name="$2"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    while [ -d "${base_dir}/${base_name}_${timestamp}" ]; do
        ((suffix++))
    done
    echo "${base_dir}/${base_name}_${timestamp}"
}

# Base directory for simulation directories
base_dir="/opt/apps/wrf_jobs"

# Prompt user for simulation name
read -p "Enter the name of the simulation: " sim_name
sim_name=$(remove_spaces "$sim_name")

# Generate simulation directory name
sim_dir=$(generate_sim_dir_name "$base_dir" "$sim_name")

# Create simulation directory
mkdir -p "$sim_dir"

# Copy "run" directory from WRF_DIR to simulation directory
WRF_DIR=$(spack location -i wrf)
if [ -d "$WRF_DIR/run" ]; then
    cp -r "$WRF_DIR/run" "$sim_dir/"
else
    echo "Error: WRF 'run' directory not found."
    exit 1
fi

cd $sim_dir/run
rm -f "wrf.exe" "tc.exe" "real.exe" "ndown.exe" "MPTABLE.TBL"

# Prompt user for namelist.input location
namelist_input=""
for ((i=0; i < 3; i++)); do
    namelist_input=$(prompt_location "Enter the location of namelist.input (Attempt $((i+1)) of 3)")
    if [ -f "$namelist_input/namelist.input" ]; then
        cp "$namelist_input" "$sim_dir/run/"
        echo "namelist.input found and copied."
        break
    else
        echo "namelist.input not found at the specified location."
        if [ $i -eq 2 ]; then
            echo "Maximum attempts reached. Exiting."
            exit 1
        fi
    fi
done

# Prompt user for "wps" directory location and check for met_em.d0* files
wps_dir=""
for ((i=0; i < 3; i++)); do
    wps_dir=$(prompt_location "Enter the location of the WPS directory (Attempt $((i+1)) of 3)")
    if [ -d "$wps_dir" ]; then
        met_files=$(find "$wps_dir" -maxdepth 1 -type f -name "met_em.d0*")
        if [ -n "$met_files" ]; then
            for met_file in $met_files; do
                cp "$met_file" "$sim_dir/run/"
            done
            echo "met_em.d0* files found and copied to 'run' directory."
            break
        else
            echo "Error: No met_em.d0* files found in the WPS directory."
            if [ $i -eq 2 ]; then
                echo "Maximum attempts reached. Exiting."
                exit 1
            fi
        fi
    else
        echo "WPS directory not found at the specified location."
        if [ $i -eq 2 ]; then
            echo "Maximum attempts reached. Exiting."
            exit 1
        fi
    fi
done

echo "Simulation directory setup completed."
echo "Go to $sim_dir/run , and run real.exe"
