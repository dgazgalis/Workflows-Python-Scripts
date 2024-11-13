import openfe
from rdkit import Chem
from openfe.utils.atommapping_network_plotting import plot_atommapping_network
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
import pathlib
import argparse

# Function to load ligands from an SDF file
def load_ligands(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, removeHs=False)
    return [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]

# Function to generate ligand network
def generate_ligand_network(ligands, mapper, scorer):
    network_planner = openfe.ligand_network_planning.generate_minimal_spanning_network
    return network_planner(ligands=ligands, mappers=[mapper], scorer=scorer)

# Function to create a ChemicalSystem
def create_chemical_system(ligand, solvent, protein=None):
    components = {'ligand': ligand, 'solvent': solvent}
    if protein:
        components['protein'] = protein
    return openfe.ChemicalSystem(components)

# Function to create transformations
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

            forward_transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping={'ligand': mapping},
                protocol=protocol,
                name=f"rbfe_{sysA.name}_{sysB.name}"
            )
            reverse_transformation = openfe.Transformation(
                stateA=sysB,
                stateB=sysA,
                mapping={'ligand': mapping},
                protocol=protocol,
                name=f"rbfe_{sysB.name}_{sysA.name}"
            )
            transformations.append(forward_transformation)
            transformations.append(reverse_transformation)
    return transformations

# Function to save transformations to disk
def save_transformations(network, output_dir):
    output_dir.mkdir(exist_ok=True)
    for transformation in network.edges:
        transformation.dump(output_dir / f"{transformation.name}.json")

# Main function to orchestrate the setup
def setup_free_energy_calculations(sdf_file_A, sdf_file_B, protein_pdb, output_dir):
    # Load ligands
    ligands_A = load_ligands(sdf_file_A)
    ligands_B = load_ligands(sdf_file_B)

    # Combine ligands
    ligands = ligands_A + ligands_B

    # Define mapper and scorer
    mapper = openfe.LomapAtomMapper(max3d=1.0, element_change=False)
    scorer = openfe.lomap_scorers.default_lomap_score

    # Generate ligand network
    ligand_network = generate_ligand_network(ligands, mapper, scorer)
    plot_atommapping_network(ligand_network)

    # Save ligand network to file
    with open("ligand_network.graphml", mode='w') as f:
        f.write(ligand_network.to_graphml())

    # Load protein and create solvent
    solvent = openfe.SolventComponent()
    protein = openfe.ProteinComponent.from_pdb_file(protein_pdb)

    # Create protocol
    protocol = RelativeHybridTopologyProtocol(RelativeHybridTopologyProtocol.default_settings())

    # Create transformations
    transformations = create_transformations(ligand_network, solvent, protein, protocol)

    # Create AlchemicalNetwork
    network = openfe.AlchemicalNetwork(transformations)

    # Save transformations to disk
    save_transformations(network, pathlib.Path(output_dir))

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Setup free energy calculations with OpenFE.")
    parser.add_argument("sdf_file_A", type=str, help="Path to the SDF file containing ligand A.")
    parser.add_argument("sdf_file_B", type=str, help="Path to the SDF file containing ligand B.")
    parser.add_argument("protein_pdb", type=str, help="Path to the PDB file containing the protein.")
    parser.add_argument("output_dir", type=str, help="Directory to save the transformations.")
    return parser.parse_args()

# Usage with command-line arguments
if __name__ == "__main__":
    args = parse_args()
    setup_free_energy_calculations(
        sdf_file_A=args.sdf_file_A,
        sdf_file_B=args.sdf_file_B,
        protein_pdb=args.protein_pdb,
        output_dir=args.output_dir
    )