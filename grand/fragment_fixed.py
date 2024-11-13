import os
import subprocess
import argparse
from openff.toolkit.topology import Molecule
from openff.interchange import Interchange
from openff.toolkit.typing.engines.smirnoff import ForceField
from lxml import etree

def run_crest(input_file, energy_window=10.0, output_dir="crest_results"):
    """
    Runs CREST on the input molecule to generate conformers and parameterize each conformer using OpenFF.
    The dihedral bond constants are set high to effectively freeze them.
    
    Parameters:
    - input_file (str): Path to the input SDF file.
    - energy_window (float): Energy window for selecting conformers (in kcal/mol).
    - output_dir (str): Directory to store the output files.
    """
    ensure_crest_installed()
    create_output_directory(output_dir)
    xyz_file, sdf_name = convert_sdf_to_xyz(input_file, output_dir)
    run_crest_command(xyz_file, energy_window)
    split_and_parametrize_conformers(output_dir, sdf_name)
    move_output_files(output_dir)

def ensure_crest_installed():
    """Ensures that CREST is installed and available in the system PATH."""
    try:
        subprocess.run(["crest", "--version"], check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: CREST is not installed or not available in your PATH.")
        sys.exit(1)

def create_output_directory(output_dir):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def convert_sdf_to_xyz(input_file, output_dir):
    """
    Converts the input SDF file to an XYZ file.
    
    Parameters:
    - input_file (str): Path to the input SDF file.
    - output_dir (str): Directory to store the output XYZ file.
    
    Returns:
    - xyz_file (str): Path to the output XYZ file.
    - sdf_name (str): Base name of the input SDF file.
    """
    molecule = Molecule.from_file(input_file)
    sdf_name = os.path.splitext(os.path.basename(input_file))[0]
    xyz_file = os.path.join(output_dir, f"{sdf_name}.xyz")
    molecule.to_file(xyz_file, file_format="XYZ")
    return xyz_file, sdf_name

def run_crest_command(xyz_file, energy_window):
    """
    Runs the CREST command to generate conformers.
    
    Parameters:
    - xyz_file (str): Path to the input XYZ file.
    - energy_window (float): Energy window for selecting conformers (in kcal/mol).
    """
    crest_command = [
        "crest",
        xyz_file,
        f"--ewin {energy_window}",
    ]
    try:
        print(f"Running CREST on {xyz_file} with an energy window of {energy_window} kcal/mol...")
        subprocess.run(crest_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: CREST command failed with error: {e}")
        sys.exit(1)

def split_and_parametrize_conformers(output_dir, sdf_name):
    """
    Splits the CREST output into individual conformers and parameterizes each using OpenFF.
    
    Parameters:
    - output_dir (str): Directory to store the conformer files.
    - sdf_name (str): Base name of the input SDF file.
    """
    conformers_file = "crest_conformers.xyz"
    if os.path.exists(conformers_file):
        with open(conformers_file, 'r') as f:
            lines = f.readlines()

        num_atoms = 0
        conformer_idx = 0
        in_conformer = False
        conformer_lines = []

        for line in lines:
            if line.strip().isdigit() and not in_conformer:
                if num_atoms > 0 and len(conformer_lines) == num_atoms + 2:
                    save_conformer_and_parametrize(conformer_lines, output_dir, sdf_name, conformer_idx)
                    conformer_idx += 1
                num_atoms = int(line.strip())
                conformer_lines = [line]
                in_conformer = True
            elif in_conformer:
                conformer_lines.append(line)
                if len(conformer_lines) == num_atoms + 2:
                    in_conformer = False

        # Write the last conformer
        if num_atoms > 0 and len(conformer_lines) == num_atoms + 2:
            save_conformer_and_parametrize(conformer_lines, output_dir, sdf_name, conformer_idx)
            print(f"Saved conformer {conformer_idx} to {os.path.join(output_dir, f'{sdf_name}_conformer_{conformer_idx}.xyz')}")

def save_conformer_and_parametrize(conformer_lines, output_dir, sdf_name, conformer_idx):
    """
    Saves a conformer to a file and parameterizes it using OpenFF.
    
    Parameters:
    - conformer_lines (list): List of lines representing the conformer.
    - output_dir (str): Directory to store the conformer file.
    - sdf_name (str): Base name of the input SDF file.
    - conformer_idx (int): Index of the conformer.
    """
    output_filename = os.path.join(output_dir, f"{sdf_name}_conformer_{conformer_idx}.xyz")
    with open(output_filename, 'w') as out_f:
        out_f.writelines(conformer_lines)
    parametrize_conformer(output_filename, output_dir, sdf_name, conformer_idx)

def parametrize_conformer(conformer_file, output_dir, sdf_name, conformer_idx):
    """
    Parameterize a conformer using OpenFF and set dihedral bond constants high to freeze them.
    Export parameters to an XML file and save a PDB representation of the conformer.
    
    Parameters:
    - conformer_file (str): Path to the conformer XYZ file.
    - output_dir (str): Directory to store the parameterized conformer.
    - sdf_name (str): Base name of the input SDF file.
    - conformer_idx (int): Index of the conformer.
    """
    # Load the molecule from the XYZ file
    molecule = Molecule.from_file(conformer_file)

    # Load the OpenFF force field
    force_field = ForceField('openff_unconstrained-2.0.0.offxml')

    # Create the OpenFF Interchange object
    interchange = Interchange.from_smirnoff(force_field, molecule.to_topology())

    # Set dihedral bond constants to a high value to freeze them
    for potential in interchange.collections['ProperTorsions'].potentials.values():
        potential.parameters['k'] = 1.0e6  # Set a very high force constant

    # Export the parameterized conformer to XML
    param_xml_file = os.path.join(output_dir, f"{sdf_name}_conformer_{conformer_idx}_params.xml")
    export_parameters_to_xml(interchange, param_xml_file)
    print(f"Parameterized conformer {conformer_idx} saved to {param_xml_file}")

    # Save PDB representation of the conformer
    pdb_file = os.path.join(output_dir, f"{sdf_name}_conformer_{conformer_idx}.pdb")
    molecule.to_file(pdb_file, file_format="PDB")
    print(f"Conformer {conformer_idx} saved as PDB to {pdb_file}")

def export_parameters_to_xml(interchange, param_xml_file):
    """
    Exports the parameters of a conformer to an XML file.
    
    Parameters:
    - interchange (Interchange): The OpenFF Interchange object containing the parameters.
    - param_xml_file (str): Path to the output XML file.
    """
    root = etree.Element("ParameterSet")
    for potential_key, potential in interchange.collections['ProperTorsions'].potentials.items():
        torsion = etree.SubElement(root, "Torsion")
        torsion.set("id", str(potential_key))
        for param_name, param_value in potential.parameters.items():
            param = etree.SubElement(torsion, param_name)
            param.text = str(param_value)
    tree = etree.ElementTree(root)
    tree.write(param_xml_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

def move_output_files(output_dir):
    """Moves the CREST output files to the specified output directory."""
    output_files = ["crest_conformers.xyz", "crest_best.xyz"]
    for file in output_files:
        if os.path.exists(file):
            os.rename(file, os.path.join(output_dir, file))
            print(f"Moved {file} to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CREST on an input molecule and parameterize the generated conformers.")
    parser.add_argument("-i, --input", type=str, help="Path to the input SDF file.")
    parser.add_argument("-e, --energy_window", type=float, default=10.0, help="Energy window for selecting conformers (in kcal/mol). Default is 10.0.")
    parser.add_argument("-o, --output_dir", type=str, default="crest_results", help="Directory to store the output files. Default is 'crest_results'.")

    args = parser.parse_args()

    # Run the script
    run_crest(args.input_file, args.energy_window, args.output_dir)
