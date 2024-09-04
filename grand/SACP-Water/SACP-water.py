import argparse
import subprocess
import numpy as np
from Bio.PDB import PDBParser, PDBIO

#Create input.pdb for SACP
def preprocess_pdb(input_pdb, output_pdb, remove_hydrogens=False):
    # Load the PDB file
    parser = PDBParser()
    structure = parser.get_structure("input", input_pdb)
    
    # Remove hydrogens if requested
    if remove_hydrogens:
        for model in structure:
            for chain in model:
                residues_to_remove = []
                for residue in chain:
                    if residue.get_id()[0] != ' ' or residue.get_id()[1] == 'H':
                        residues_to_remove.append(residue.get_id())
                for res_id in residues_to_remove:
                    chain.detach_child(res_id)
    
    # Remove ligands
    for model in structure:
        for chain in model:
            residues_to_remove = []
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    residues_to_remove.append(residue.get_id())
            for res_id in residues_to_remove:
                chain.detach_child(res_id)
    
    # Save the intermediate PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save("intermediate.pdb")
    
    # Run pdb4amber
    subprocess.run(["pdb4amber", "-i", "intermediate.pdb", "-o", "pdb4amber_output.pdb"])
    
    # Run tleap
    with open("tleap.in", "w") as f:
        f.write("""source leaprc.protein.ff14SB
        source leaprc.water.tip3p
        source leaprc.gaff2
        pdb = loadpdb pdb4amber_output.pdb
        savepdb pdb {}
        quit""".format(output_pdb))
    
    subprocess.run(["tleap", "-f", "tleap.in"])


#Coarse annealing functions
def run_annealing_coarse(schedule):
    # First, run initialize.py to generate the uvt_initial system
    subprocess.run(["python", "initialize.py"])
    
    previous_pdb = "uvt_initial.pdb"
    
    for adams_value in schedule:
        # Format the float to 1 decimal place
        adams_value_str = f"{adams_value:.1f}"
        print(f"Running with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
    
        # Run the annealing.py script with the specified parameters
        subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "coarse"])
        
        # Update previous_pdb for the next iteration
        previous_pdb = f"uvt_{adams_value_str}.pdb"

def continue_annealing_coarse(adams_value, previous_pdb):
    # Format the float to 1 decimal place
    adams_value_str = f"{adams_value:.1f}"    
    print(f"Running with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
    # Run the annealing.py script with the specified parameters
    subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "coarse"])
    
#Fine annealing Function 
def continue_annealing_fine(adams_value, previous_pdb):
    # Format the float to 1 decimal place
    adams_value_str = f"{adams_value:.1f}"
    print(f"Running fine annealing with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
    # Run the annealing.py script with the specified parameters
    subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "fine"])

#Post process                       
def count_non_protein_molecules(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    
    non_protein_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # Check if it's a hetero-atom (non-protein)
                    non_protein_count += 1
    
    return non_protein_count

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a PDB file and run annealing.")
    parser.add_argument('-i', '--input', required=True, help="Input PDB file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    input_pdb = args.input  # User-provided PDB file via command-line argument
    output_pdb = "input.pdb"  # Output PDB file
    remove_hydrogens = True  # Set to True if hydrogens should be removed
    
    # Preprocess the PDB file
    preprocess_pdb(input_pdb, output_pdb, remove_hydrogens)
    
    # Define the initial annealing schedule
    initial_schedule = np.arange(15, -18, -3).tolist()
    
    # Print the initial annealing schedule for user information
    print("Initial Annealing Schedule:", initial_schedule)
    
    # Dictionary to store non-protein molecule counts
    molecule_counts = {}

    # Run the initial annealing process
    for adams_value in initial_schedule:
        run_annealing_coarse([adams_value])
        pdb_file = f"uvt_{adams_value:.1f}.pdb"
        count = count_non_protein_molecules(pdb_file)
        molecule_counts[adams_value] = count
        print(f"Adams value: {adams_value:.1f}, Non-protein molecules: {count}")

    # Continue annealing if necessary
    if molecule_counts[initial_schedule[-1]] > 0:
        current_adams_value = initial_schedule[-1]
        while True:
            next_adams_value = current_adams_value - 3
            continue_annealing_coarse(next_adams_value, f"uvt_{current_adams_value:.1f}.pdb")
            
            current_pdb = f"uvt_{next_adams_value:.1f}.pdb"
            count = count_non_protein_molecules(current_pdb)
            molecule_counts[next_adams_value] = count
            print(f"Adams value: {next_adams_value:.1f}, Non-protein molecules: {count}")
            
            if count == 0:
                break
            
            current_adams_value = next_adams_value

    # Find the maximum count and the corresponding Adams value
    max_count = max(molecule_counts.values())
    max_adams = max(molecule_counts, key=molecule_counts.get)

    # Find the Adams value with count closest to half of the maximum,
    # then round down to the next Adams value on the schedule
    half_max = max_count / 2
    half_max_adams = min(molecule_counts, key=lambda x: abs(molecule_counts[x] - half_max))
    
    # Get all Adams values in descending order
    adams_values = sorted(molecule_counts.keys(), reverse=True)
    
    # Find the next lower Adams value on the schedule
    half_max_adams_rounded = next(value for value in adams_values if value <= half_max_adams)

    print(f"Maximum non-protein molecule count: {max_count} at Adams value: {max_adams:.1f}")
    print(f"Half of maximum count: {half_max}")
    print(f"Adams value with count closest to half of maximum: {half_max_adams:.1f} (count: {molecule_counts[half_max_adams]})")
    print(f"Rounded down Adams value: {half_max_adams_rounded:.1f} (count: {molecule_counts[half_max_adams_rounded]})")

    # Create and run the second annealing schedule
    print("\nStarting second annealing schedule:")
    second_schedule = np.arange(half_max_adams_rounded, -18, -1).tolist()
    
    print("Second Annealing Schedule:", second_schedule)

    for adams_value in second_schedule:
        if adams_value not in molecule_counts:
            run_annealing_coarse([adams_value])
            pdb_file = f"uvt_{adams_value:.1f}.pdb"
            count = count_non_protein_molecules(pdb_file)
            molecule_counts[adams_value] = count
        else:
            count = molecule_counts[adams_value]
        
        print(f"Adams value: {adams_value:.1f}, Non-protein molecules: {count}")

        if count == 0:
            break

    # Print summary of annealing process
    print("\nComplete Annealing Process Summary:")
    for adams_value, count in sorted(molecule_counts.items(), reverse=True):
        print(f"Adams value: {adams_value:.1f}, Non-protein molecules: {count}")

    if molecule_counts[min(molecule_counts.keys())] == 0:
        print("\nAnnealing completed successfully. All non-protein molecules removed.")
    else:
        print("\nAnnealing completed, but some non-protein molecules remain.")
        print(f"Final Adams value: {min(molecule_counts.keys()):.1f}")
        print(f"Remaining non-protein molecules: {molecule_counts[min(molecule_counts.keys())]}")

if __name__ == "__main__":
    main()