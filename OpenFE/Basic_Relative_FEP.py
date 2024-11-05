import os
import argparse
import pathlib
import json
from rdkit import Chem
import itertools
import openfe
from openfe.setup import LigandAtomMapping
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
from openfe import ProteinComponent, SmallMoleculeComponent

def read_ligands_from_sdf(sdf_file):
    ligands = []
    mol_supplier = Chem.SDMolSupplier(sdf_file)
    for mol in mol_supplier:
        if mol is not None:
            ligand = SmallMoleculeComponent.from_rdkit(mol)
            ligands.append(ligand)
    return ligands

def load_protocol_settings(json_file):
    with open(json_file, 'r') as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError:
            print("Error: The settings file is empty or contains invalid JSON.")
            exit(1)
    return settings

def create_protocol_settings():
    protocol_settings = RelativeHybridTopologyProtocol.default_settings()

    # Explicitly set the default values for easy editing
    protocol_settings.forcefield_settings.constraints = 'hbonds'
    protocol_settings.forcefield_settings.rigid_water = True
    protocol_settings.forcefield_settings.hydrogen_mass = 3.0  # Remove unit.amu
    protocol_settings.forcefield_settings.forcefields = [
        'amber/ff14SB.xml',
        'amber/tip3p_standard.xml',
        'amber/tip3p_HFE_multivalent.xml',
        'amber/phosaa10.xml'
    ]
    protocol_settings.forcefield_settings.small_molecule_forcefield = 'openff-2.1.0'
    protocol_settings.forcefield_settings.nonbonded_cutoff = 1.0 * unit.nanometer
    protocol_settings.forcefield_settings.nonbonded_method = 'PME'

    protocol_settings.thermo_settings.temperature = 298.15 * unit.kelvin
    protocol_settings.thermo_settings.pressure = 1.0 * unit.atmosphere

    protocol_settings.protocol_repeats = 3

    protocol_settings.solvation_settings.solvent_model = 'tip3p'
    protocol_settings.solvation_settings.solvent_padding = 1.2 * unit.nanometer

    protocol_settings.partial_charge_settings.partial_charge_method = 'am1bcc'

    protocol_settings.lambda_settings.lambda_windows = 11

    protocol_settings.alchemical_settings.softcore_LJ = 'gapsys'
    protocol_settings.alchemical_settings.softcore_alpha = 0.85

    protocol_settings.simulation_settings.equilibration_length = 1.0 * unit.nanosecond
    protocol_settings.simulation_settings.production_length = 5.0 * unit.nanosecond
    protocol_settings.simulation_settings.n_replicas = 11

    protocol_settings.output_settings.checkpoint_interval = 250 * unit.picosecond
    protocol_settings.output_settings.output_indices = 'not water'

    # Write the settings to a JSON file
    with open('protocol_settings.json', 'w') as f:
        json.dump(protocol_settings.__dict__, f, indent=4, default=str)

    return protocol_settings

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform FEP calculations using OpenFE")
    parser.add_argument("-p", "--protein", required=True, help="Path to the protein structure file (PDB format)")
    parser.add_argument("-l", "--ligands", required=True, help="Path to the ligands file (SDF format)")
    parser.add_argument("-s", "--settings", default="protocol_settings.json", help="Path to the protocol settings JSON file")
    args = parser.parse_args()

    # Create protocol settings and write to JSON file
    protocol_settings = create_protocol_settings()
    print("Protocol settings have been created and written to 'protocol_settings.json'.")

    # Load protocol settings
    settings = load_protocol_settings(args.settings)

    # Read protein structure
    protein = ProteinComponent.from_pdb_file(args.protein)

    # Read ligands from SDF file
    supp = Chem.SDMolSupplier(args.ligands, removeHs=False)
    ligands = [SmallMoleculeComponent.from_rdkit(mol) for mol in supp if mol is not None]

    # Set up LOMAP scorer and network mapper
    mapper = openfe.LomapAtomMapper(max3d=1.0, element_change=False)
    scorer = openfe.lomap_scorers.default_lomap_score
    network_planner = openfe.ligand_network_planning.generate_minimal_spanning_network

    # Create the FEP protocol
    protocol = RelativeHybridTopologyProtocol(settings=protocol_settings)

    # Create default solvent (water with NaCl at 0.15 M)
    solvent = openfe.SolventComponent()

    # Create network for all ligands
    ligand_network = network_planner(
        ligands=ligands,
        mappers=[mapper],
        scorer=scorer
    )
    
    # Save the ligand network to a GraphML file
    with open("ligand_network.graphml", mode='w') as f:
        f.write(ligand_network.to_graphml())

    # Create transformations for both solvent and complex legs
    transformations = []
    for edge in ligand_network.edges:
        for leg in ['solvent', 'complex']:
            # Use the solvent and protein created above
            sysA_dict = {'ligand': edge.componentA, 'solvent': solvent}
            sysB_dict = {'ligand': edge.componentB, 'solvent': solvent}
            
            if leg == 'complex':
                sysA_dict['protein'] = protein
                sysB_dict['protein'] = protein
            
            # Create named ChemicalSystem objects
            sysA = openfe.ChemicalSystem(sysA_dict, name=f"{edge.componentA.name}_{leg}")
            sysB = openfe.ChemicalSystem(sysB_dict, name=f"{edge.componentB.name}_{leg}")
            
            prefix = "easy_rbfe_"  # prefix to exactly reproduce CLI
            
            # Create the transformation
            transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping=edge,
                protocol=protocol,
                name=f"{prefix}{sysA.name}_{sysB.name}"
            )
            
            transformations.append(transformation)

    # Create ProtocolDAGs for each transformation
    transformation_dags = []
    for transformation in transformations:
        transformation_dag = transformation.create()
        transformation_dags.append(transformation_dag)

    # Create directory for transformations
    transformation_dir = pathlib.Path("transformations")
    transformation_dir.mkdir(exist_ok=True)

    # Write out each transformation and create subfolders
    for transformation in transformations:
        subfolder_name = f"{transformation.stateA.name}_{transformation.stateB.name}"
        subfolder_path = transformation_dir / subfolder_name
        subfolder_path.mkdir(exist_ok=True)
        
        transformation.dump(subfolder_path / f"{transformation.name}.json")

    print(f"Transformations have been written to subfolders in the '{transformation_dir}' directory.")

    # Perform FEP calculations for all transformation DAGs
    results = []
    for transformation, dag in zip(transformations, transformation_dags):
        subfolder_name = f"{transformation.stateA.name}_{transformation.stateB.name}"
        subfolder_path = transformation_dir / subfolder_name
        
        print(f"Calculating FEP for transformation: {transformation.name}")
        print(f"Working directory: {subfolder_path}")
        
        # Change to the subfolder
        os.chdir(subfolder_path)
        
        try:
            # Dry run to ensure the hybrid system can be constructed without any issues
            unit = list(dag.protocol_units)[0]
            unit.run(verbose=True)

            # Execute all protocol units
            dags = list(dag.protocol_units)
            for unit in dags:
                unit.execute(context=None)
            
            # Retrieve estimate and uncertainty if available
            estimate = dag.get_estimate()
            uncertainty = dag.get_uncertainty()
            print(f"Result: {estimate} Â± {uncertainty}")
            
            results.append((transformation.name, estimate, uncertainty))
        except Exception as e:
            print(f"Error executing transformation {transformation.name}: {str(e)}")
        
        # Change back to the original directory
        os.chdir("../../")

    # Print the results
    for transformation_name, estimate, uncertainty in results:
        print(f"FEP result for {transformation_name}:")
        print(f"Estimate: {estimate} Â± {uncertainty}")
        print("---")

if __name__ == "__main__":
    main()
