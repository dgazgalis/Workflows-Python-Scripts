import openfe
from openfe.setup import LomapAtomMapper
from openfe.setup.ligand_network_planning import generate_lomap_network
from openfe import SolventComponent, ProteinComponent
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
import pathlib
import argparse
from rdkit import Chem

# Function to load ligands from an SDF file
def load_ligands(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, removeHs=False)
    return [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]

# Function to generate ligand network and create transformations
def generate_ligand_network_and_transformations(ligands, solvent, protein, protocol, output_dir):
    # Define mapper and scorer
    mapper = LomapAtomMapper(max3d=2.0, element_change=False)
    scorer = openfe.lomap_scorers.default_lomap_score
    
    # Generate ligand network
    ligand_network = generate_lomap_network(molecules=ligands, scorer=scorer, mappers=[mapper])

    # Save ligand network to file
    with open("ligand_network.graphml", mode='w') as f:
        f.write(ligand_network.to_graphml())

    # Create transformations
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

            # Ensure the ligands are correctly mapped
            relevant_edge = [edge for edge in ligand_network.edges if edge.componentB.name == mapping.componentB.name][0]

            forward_transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping={'ligand': relevant_edge},
                protocol=protocol,
                name=f"rbfe_{sysA.name}_{sysB.name}"
            )
            reverse_transformation = openfe.Transformation(
                stateA=sysB,
                stateB=sysA,
                mapping={'ligand': relevant_edge},
                protocol=protocol,
                name=f"rbfe_{sysB.name}_{sysA.name}"
            )
            transformations.append(forward_transformation)
            transformations.append(reverse_transformation)

    # Create AlchemicalNetwork
    network = openfe.AlchemicalNetwork(transformations)

    # Save transformations to disk
    save_transformations(network, pathlib.Path(output_dir))

    return network, transformations

# Function to save transformations to disk
def save_transformations(network, output_dir):
    output_dir.mkdir(exist_ok=True)
    for transformation in network.edges:
        transformation.dump(output_dir / f"{transformation.name}.json")

# Function to create and run ProtocolDAG for a single transformation
def create_and_run_protocol_dag(transformation):
    # Create ProtocolDAG
    protocol_dag = transformation.create()

    # Run the first protocol unit
    protocol_unit = list(protocol_dag.protocol_units)[0]
    protocol_unit.run(dry=True, verbose=True)

# Function to execute the ProtocolDAG
def execute_DAG(dag, scratch_basedir, shared_basedir):
    return dag.run(scratch_basedir=scratch_basedir, shared_basedir=shared_basedir)

# Main function to orchestrate the setup
def setup_free_energy_calculations(sdf_file_A, sdf_file_B, protein_pdb, output_dir):
    # Load ligands
    ligands_A = load_ligands(sdf_file_A)
    ligands_B = load_ligands(sdf_file_B)

    # Combine ligands
    ligands = ligands_A + ligands_B

    # Load protein and create solvent
    solvent = SolventComponent(positive_ion='Na', negative_ion='Cl', neutralize=True, ion_concentration=0.15*unit.molar)
    protein = ProteinComponent.from_pdb_file(protein_pdb)

    # Create protocol
    protocol = RelativeHybridTopologyProtocol(RelativeHybridTopologyProtocol.default_settings())

    # Generate ligand network and create transformations
    network, transformations = generate_ligand_network_and_transformations(ligands, solvent, protein, protocol, output_dir)

    # Create and run ProtocolDAG for each transformation
    for transformation in transformations:
        create_and_run_protocol_dag(transformation)

    # Run the simulations
    complex_path = pathlib.Path('./complex')
    complex_path.mkdir(exist_ok=True)
    solvent_path = pathlib.Path('./solvent')
    solvent_path.mkdir(exist_ok=True)

    # First the complex transformation
    complex_dag_results = execute_DAG(network, scratch_basedir=complex_path, shared_basedir=complex_path)

    # Next the solvent state transformation
    solvent_dag_results = execute_DAG(network, scratch_basedir=solvent_path, shared_basedir=solvent_path)

    # Get the complex and solvent results
    complex_results = protocol.gather([complex_dag_results])
    solvent_results = protocol.gather([solvent_dag_results])

    # Print the results
    print(f"Complex dG: {complex_results.get_estimate()}, err {complex_results.get_uncertainty()}")
    print(f"Solvent dG: {solvent_results.get_estimate()}, err {solvent_results.get_uncertainty()}")

# Function to construct default output directory name
def get_default_output_dir(sdf_file_A, sdf_file_B, protein_pdb):
    ligand_A_name = pathlib.Path(sdf_file_A).stem
    ligand_B_name = pathlib.Path(sdf_file_B).stem
    protein_name = pathlib.Path(protein_pdb).stem
    return f"{ligand_A_name}_{ligand_B_name}_{protein_name}"

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Setup free energy calculations with OpenFE.")
    parser.add_argument("-l1", "--sdf_file_A", type=str, required=True, help="Path to the SDF file containing ligand A.")
    parser.add_argument("-l2", "--sdf_file_B", type=str, required=True, help="Path to the SDF file containing ligand B.")
    parser.add_argument("-p", "--protein_pdb", type=str, required=True, help="Path to the PDB file containing the protein.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Directory to save the transformations.")
    return parser.parse_args()

# Usage with command-line arguments
if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.sdf_file_A, args.sdf_file_B, args.protein_pdb)
    setup_free_energy_calculations(
        sdf_file_A=args.sdf_file_A,
        sdf_file_B=args.sdf_file_B,
        protein_pdb=args.protein_pdb,
        output_dir=args.output_dir
    )

