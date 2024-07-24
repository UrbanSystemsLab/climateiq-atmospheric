#!/bin/bash

# set env variable
WPS_DIR=`spack location -i wps`

echo "This script will setup a directory for WPS jobs and then invoke geogrid"

# Initialize retry count
retryCount=0

# Loop for three retries
while [ $retryCount -lt 3 ]; do
    # Prompt the user for input
    echo "Enter the similation name, no spaces please:"
    read userInput

    # Check if the input is empty
    if [ -z "$userInput" ]; then
        echo "You entered nothing."

        # Increment the retry count
        ((retryCount++))

        # Check if it's the last retry
        if [ $retryCount -eq 3 ]; then
            echo "No input provided after 3 retries. Exiting."
            exit 1
        else
            echo "Please try again."
        fi
    else
        echo "You entered: $userInput"
        break
    fi
done


# Remove spaces from the input
userInput=$(echo "$userInput" | tr -d ' ')

echo "Creating job directory for this simulation"

# Get today's date in yyyymmdd format
today=$(date +"%Y%m%d")

# Construct the directory name with user input and today's date
directoryName="/opt/apps/wps_jobs/${userInput}_${today}"

# Check if the directory already exists
if [ -d "$directoryName" ]; then
    # If the directory already exists, find a suitable name by appending _number
    i=1
    while [ -d "${directoryName}_${i}" ]; do
        ((i++))
    done
    directoryName="${directoryName}_${i}"
fi

# Create the directory
mkdir "$directoryName"

echo "Directory '$directoryName' created."

# Setup WPS geogrid pre-requirements

echo " Setting up directory structure now"

cd $directoryName

attempts=0

while [ $attempts -lt 3 ]; do
    echo "Enter the directory containing namelist.wps:"
    read -r namelistDir

    # Check if namelist.input exists in the directory
    if [ -f "$namelistDir/namelist.wps" ]; then
        # Copy the namelist.input file
        cp "$namelistDir/namelist.wps" .

        # Inform user and exit the loop
        echo "namelist.wps copied successfully."
        break
    else
        echo "Error: namelist.wps not found in the specified directory."
        attempts=$((attempts + 1))
    fi
done

# If the loop ends without success, inform the user and exit
if [ $attempts -eq 3 ]; then
    echo "Error: namelist.wps not found after 3 attempts. Exiting."
fi

cp -Rf $WPS_DIR/geogrid/ .
cp -Rf $WPS_DIR/ungrib/ .
cp -Rf $WPS_DIR/metgrid/ .
cp $WPS_DIR/link_grib.csh .
ln -s ungrib/Variable_Tables/Vtable.GFS Vtable

# Initialize attempt counter
fnl_attempts=0

# Loop for up to 3 attempts
while [ $fnl_attempts -lt 3 ]; do
    echo "Enter the 'fnl' directory:"
    read -r userInput2

    # Check if the input is empty
    if [ -z "$userInput2" ]; then
        echo "You entered nothing."
    else
        mkdir -p fnl  # Create fnl directory if it doesn't exist
        cp "$userInput2"/fnl* fnl/  # Move files starting with 'fnl' into fnl directory

        # Check if any files starting with 'fnl' were moved
        if [ -n "$(ls -A fnl)" ]; then
            echo "Files starting with 'fnl' moved successfully."
            break  # Exit the loop
        else
            echo "No files starting with 'fnl' found in the specified directory."
            fnl_attempts=$((fnl_attempts + 1))
        fi
    fi
done

# If the loop ends without success, inform the user and exit
if [ $fnl_attempts -eq 3 ]; then
    echo "Error: Files starting with 'fnl' not found after 3 attempts. Exiting."
fi

echo "Directory creation is done, you can go to the directory $directoryName now and execute geogrid.exe"
