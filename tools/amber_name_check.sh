#!/bin/bash

# Check if a PDB file is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_pdb_file>"
    exit 1
fi

input_pdb="$1"
base_name=$(basename "$input_pdb" .pdb)

# Step 1: Run pdb4amber to clean up the PDB file
echo "Running pdb4amber..."
pdb4amber -i "$input_pdb" -o "${base_name}_clean.pdb" --nohyd

# Step 2: Create a tleap input file
cat << EOF > check_amber.in
source leaprc.protein.ff14SB
source leaprc.water.tip3p

mol = loadpdb ${base_name}_clean.pdb
check mol
saveamberparm mol ${base_name}.prmtop ${base_name}.inpcrd
savepdb mol ${base_name}_amber.pdb
quit
EOF

# Step 3: Run tleap
echo "Running tleap..."
tleap -f check_amber.in > tleap_output.log

# Step 4: Check for errors and outputs
if grep -q "ERROR" tleap_output.log; then
    echo "Errors found in tleap output. Please check tleap_output.log for details."
else
    if [ -f "${base_name}_amber.pdb" ]; then
        echo "No errors found. Your PDB file is compatible with AMBER14 force field."
        echo "Generated files:"
        echo "  - ${base_name}.prmtop (topology file)"
        echo "  - ${base_name}.inpcrd (coordinate file)"
        echo "  - ${base_name}_amber.pdb (AMBER-compatible PDB for simulations)"
    else
        echo "No errors found, but failed to generate the AMBER-compatible PDB file."
        echo "Please check tleap_output.log for details."
    fi
fi

# Optional: Clean up intermediate files
# rm check_amber.in ${base_name}_clean.pdb

echo "Script completed. Please review tleap_output.log for full details."