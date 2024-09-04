import argparse
import subprocess
import numpy as np
from Bio.PDB import PDBParser, PDBIO

def preprocess_pdb(input_pdb, output_pdb, remove_hydrogens=False):
    """
    Preprocess the PDB file by removing hydrogens and ligands, and running pdb4amber and tleap.
    """
    parser = PDBParser()
    structure = parser.get_structure("input", input_pdb)
    
    if remove_hydrogens:
        for model in structure:
            for chain in model:
                residues_to_remove = [residue.get_id() for residue in chain if residue.get_id()[0] != ' ' or residue.get_id()[1] == 'H']
                for res_id in residues_to_remove:
                    chain.detach_child(res_id)
    
    for model in structure:
        for chain in model:
            residues_to_remove = [residue.get_id() for residue in chain if residue.get_id()[0] != ' ']
            for res_id in residues_to_remove:
                chain.detach_child(res_id)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save("intermediate.pdb")
    
    subprocess.run(["pdb4amber", "-i", "intermediate.pdb", "-o", "pdb4amber_output.pdb"])
    
    with open("tleap.in", "w") as f:
        f.write("""source leaprc.protein.ff14SB
        source leaprc.water.tip3p
        source leaprc.gaff2
        pdb = loadpdb pdb4amber_output.pdb
        savepdb pdb {}
        quit""".format(output_pdb))
    
    subprocess.run(["tleap", "-f", "tleap.in"])

def run_annealing_coarse(schedule):
    """
    Run coarse annealing with the given schedule.
    """
    subprocess.run(["python", "initialize.py"])
    
    previous_pdb = "uvt_initial.pdb"
    
    for adams_value in schedule:
        adams_value_str = f"{adams_value:.1f}"
        print(f"Running with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
        subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "coarse"])
        previous_pdb = f"uvt_{adams_value_str}.pdb"

def continue_annealing_coarse(adams_value, previous_pdb):
    """
    Continue coarse annealing with the given Adams value and previous PDB.
    """
    adams_value_str = f"{adams_value:.1f}"
    print(f"Running with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
    subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "coarse"])

def continue_annealing_fine(adams_value, previous_pdb):
    """
    Continue fine annealing with the given Adams value and previous PDB.
    """
    adams_value_str = f"{adams_value:.1f}"
    print(f"Running fine annealing with Adams value: {adams_value_str}, Previous PDB: {previous_pdb}")
    subprocess.run(["python", "annealing.py", "-a", adams_value_str, "-p", previous_pdb, "-m", "fine"])

def count_non_protein_molecules(pdb_file):
    """
    Count non-protein molecules in the given PDB file.
    """
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    
    non_protein_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    non_protein_count += 1
    
    return non_protein_count

def main():
    """
    Main function to parse command-line arguments, preprocess the PDB file, and run the annealing process.
    """
    parser = argparse.ArgumentParser(description="Process a PDB file and run annealing.")
    parser.add_argument('-i', '--input', required=True, help="Input PDB file")
    args = parser.parse_args()
    
    input_pdb = args.input
    output_pdb = "input.pdb"
    remove_hydrogens = True
    
    preprocess_pdb(input_pdb, output_pdb, remove_hydrogens)
    
    initial_schedule = np.arange(15, -18, -3).tolist()
    print("Initial Annealing Schedule:", initial_schedule)
    
    molecule_counts = {}

    for adams_value in initial_schedule:
        run_annealing_coarse([adams_value])
        pdb_file = f"uvt_{adams_value:.1f}.pdb"
        count = count_non_protein_molecules(pdb_file)
        molecule_counts[adams_value] = count
        print(f"Adams value: {adams_value:.1f}, Non-protein molecules: {count}")

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

    max_count = max(molecule_counts.values())
    max_adams = max(molecule_counts, key=molecule_counts.get)

    half_max = max_count / 2
    half_max_adams = min(molecule_counts, key=lambda x: abs(molecule_counts[x] - half_max))
    
    adams_values = sorted(molecule_counts.keys(), reverse=True)
    half_max_adams_rounded = next(value for value in adams_values if value <= half_max_adams)

    print(f"Maximum non-protein molecule count: {max_count} at Adams value: {max_adams:.1f}")
    print(f"Half of maximum count: {half_max}")
    print(f"Adams value with count closest to half of maximum: {half_max_adams:.1f} (count: {molecule_counts[half_max_adams]})")
    print(f"Rounded down Adams value: {half_max_adams_rounded:.1f} (count: {molecule_counts[half_max_adams_rounded]})")

    print("\nStarting second annealing schedule:")
    second_schedule = np.arange(half_max_adams_rounded, -20, -1).tolist()
    
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