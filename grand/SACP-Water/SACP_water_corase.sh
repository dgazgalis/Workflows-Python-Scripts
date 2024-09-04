#!/bin/bash

for i in $(seq 9 -3.0 -15)
do
    # Use printf to format the float to 1 decimal place
    adams_value=$(printf "%.1f" $i)
    
    # Determine the previous PDB file
    if (( $(echo "$i > 8" | bc -l) )); then
        previous_pdb="uvt_inital.pdb"
    else
        # Calculate the previous Adams value
        prev_adams=$(printf "%.1f" $(echo "$i + 3.0" | bc))
        previous_pdb="uvt_${prev_adams}.pdb"
    fi
    
    echo "Running with Adams value: $adams_value, Previous PDB: $previous_pdb"
    python annealing.py -a $adams_value -p $previous_pdb -m coarse
done
: '
for i in $(seq -4.0 -1.0 -5.0)
do
    # Use printf to format the float to 1 decimal place
    adams_value=$(printf "%.1f" $i)
    
    # Determine the previous PDB file
    if (( $(echo "$i > -5.0" | bc -l) )); then
        previous_pdb="uvt_-3.0.pdb"
    else
        # Calculate the previous Adams value
        prev_adams=$(printf "%.1f" $(echo "$i + 1.0" | bc))
        previous_pdb="uvt_${prev_adams}.pdb"
    fi
    
    echo "Running with Adams value: $adams_value, Previous PDB: $previous_pdb"
    python annealing.py -a $adams_value -p $previous_pdb
done


for i in $(seq -5.1 -0.1 -6.0)
do
    # Use printf to format the float to 1 decimal place
    adams_value=$(printf "%.1f" $i)
    
    # Determine the previous PDB file
    if (( $(echo "$i > -5.0" | bc -l) )); then
        previous_pdb="uvt_-5.0.pdb"
    else
        # Calculate the previous Adams value
        prev_adams=$(printf "%.1f" $(echo "$i + 0.1| bc))
        previous_pdb="uvt_${prev_adams}.pdb"
    fi
    
    echo "Running with Adams value: $adams_value, Previous PDB: $previous_pdb"
    python annealing.py -a $adams_value -p $previous_pdb
done
'

