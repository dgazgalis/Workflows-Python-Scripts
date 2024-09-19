import argparse
import openfe
from rdkit import Chem
from openfe.utils.atommapping_network_plotting import plot_atommapping_network
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
import pathlib
import subprocess

# Load ligands using RDKit
def load_ligands_rdkit(input_file):
    supp = Chem.SDMolSupplier(input_file, removeHs=False)
    return [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]

# Generate ligand network
def generate_ligand_network(ligands):
    mapper = openfe.LomapAtomMapper(max3d=1.0, element_change=False)
    scorer = openfe.lomap_scorers.default_lomap_score
    network_planner = openfe.ligand_network_planning.generate_minimal_spanning_network
    
    return network_planner(
        ligands=ligands,
        mappers=[mapper],
        scorer=scorer
    )

# Create chemical systems
def create_chemical_systems(mapping, solvent, protein):
    systemA = openfe.ChemicalSystem({
        'ligand': mapping.componentA,
        'solvent': solvent,
        'protein': protein
    })
    systemB = openfe.ChemicalSystem({
        'ligand': mapping.componentB,
        'solvent': solvent,
        'protein': protein
    })
    return systemA, systemB

# Create transformations
def create_transformations(ligand_network, solvent, protein, protocol):
    transformations = []
    for mapping in ligand_network.edges:
        for leg in ['solvent', 'complex']:
            sysA_dict = {'ligand': mapping.componentA, 'solvent': solvent}
            sysB_dict = {'ligand': mapping.componentB, 'solvent': solvent}

            if leg == 'complex':
                sysA_dict['protein'] = protein
                sysB_dict['protein'] = protein

            sysA = openfe.ChemicalSystem(sysA_dict, name=f"{mapping.componentA.name}_{leg}")
            sysB = openfe.ChemicalSystem(sysB_dict, name=f"{mapping.componentB.name}_{leg}")

            prefix = "rbfe_"

            transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping={'ligand': mapping},
                protocol=protocol,
                name=f"{prefix}{sysA.name}_{sysB.name}"
            )
            transformations.append(transformation)

    return transformations

# Save transformations to disk
def save_transformations(transformations, output_dir):
    transformation_dir = pathlib.Path(output_dir) / "transformations"
    transformation_dir.mkdir(parents=True, exist_ok=True)

    for transformation in transformations:
        transformation.dump(transformation_dir / f"{transformation.name}.json")
def main():
    parser = argparse.ArgumentParser(description="Basic Relative Free Energy Calculations using OpenFE.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input SDF file containing ligands.")
    parser.add_argument('-p', '--protein', type=str, required=True, help="Path to the protein PDB file.")
    parser.add_argument('-o', '--output', type=str, default=".", help="Output directory for transformation files.")
    args = parser.parse_args()

    # Load ligands using RDKit
    ligands = load_ligands_rdkit(args.input)
    
    # Generate ligand network
    ligand_network = generate_ligand_network(ligands)

    # Create solvent and protein components
    solvent = openfe.SolventComponent()
    protein = openfe.ProteinComponent.from_pdb_file(args.protein)
      
    # Set up the protocol for transformations
    default_settings = RelativeHybridTopologyProtocol.default_settings()
    default_settings.thermo_settings.temperature = 310.0 * unit.kelvin
    protocol = RelativeHybridTopologyProtocol(default_settings)

    # Create transformations
    transformations = create_transformations(ligand_network, solvent, protein, protocol)
    
    # Save transformations to disk
    save_transformations(transformations, args.output)

    # List all possible transformations
    print("Available transformations:")
    for i, transformation in enumerate(transformations):
        print(f"{i + 1}: {transformation.name}")

    # Prompt user to select a transformation
    while True:
        try:
            selection = int(input("Select a transformation to run (enter the number): "))
            if 1 <= selection <= len(transformations):
                selected_transformation = transformations[selection - 1]
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Define the working directory and results file
    working_directory = pathlib.Path(args.output) / "working-directory"
    results_file = pathlib.Path(args.output) / "results.json"

    # Ensure the working directory exists
    working_directory.mkdir(parents=True, exist_ok=True)

    # Run the selected transformation using OpenFE CLI via subprocess
    transformation_file = pathlib.Path(args.output) / "transformations" / f"{selected_transformation.name}.json"
    command = [
        "openfe", "quickrun", str(transformation_file),
        "-o", str(results_file),
        "-d", str(working_directory)
    ]
    subprocess.run(command)

    # Analyze the results using openfe gather
    results_dir = pathlib.Path(args.output) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    final_results_file = pathlib.Path(args.output) / "final_results.tsv"

    gather_command = [
        "openfe", "gather", str(results_dir),
        "--report", "dg",
        "-o", str(final_results_file)
    ]
    subprocess.run(gather_command)

    print(f"Results have been analyzed and saved to {final_results_file}")

if __name__ == "__main__":
    main()