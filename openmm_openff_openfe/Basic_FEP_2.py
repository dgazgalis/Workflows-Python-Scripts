import argparse
import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from openfe import SmallMoleculeComponent, SolventComponent, ProteinComponent, ChemicalSystem, Transformation
from openfe.setup import LomapAtomMapper
from openfe.setup.ligand_network_planning import generate_lomap_network
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openff.units import unit
from openff.toolkit import Molecule

def main():
    parser = argparse.ArgumentParser(description="Perform Free Energy Perturbation (FEP) using OpenMM, OpenFF, and OpenFE.")
    parser.add_argument('-l', '--ligand', required=True, help='Path to the ligand SDF file')
    parser.add_argument('-p', '--protein', required=True, help='Path to the protein PDB file')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory')
    args = parser.parse_args()

    # Extract the contents of the sdf file and visualize it
    ligands_rdmol = [mol for mol in Chem.SDMolSupplier(args.ligand, removeHs=False)]

    for ligand in ligands_rdmol:
        AllChem.Compute2DCoords(ligand)

    Chem.Draw.MolsToGridImage(ligands_rdmol)

    # Load ligands using RDKit
    ligands_sdf = Chem.SDMolSupplier(args.ligand, removeHs=False)
    ligand_mols = [SmallMoleculeComponent(sdf) for sdf in ligands_sdf]

    # Load ligands using OpenFF toolkit
    ligands_sdf = Molecule.from_file(args.ligand)
    ligand_mols = [SmallMoleculeComponent.from_openff(sdf) for sdf in ligands_sdf]

    # Create a LOMAP network
    mapper = LomapAtomMapper()
    lomap_network = generate_lomap_network(
        molecules=ligand_mols,
        scorer=openfe.lomap_scorers.default_lomap_score,
        mappers=[LomapAtomMapper(),]
    )

    lomap_edges = [edge for edge in lomap_network.edges]

    # Pick an edge
    edge = lomap_edges[1]

    # Print the smiles of the molecules and the mapping
    print("molecule A smiles: ", edge.componentA.smiles)
    print("molecule B smiles: ", edge.componentB.smiles)
    print("map between molecule A and B: ", edge.componentA_to_componentB)

    with open(f"{args.output}/network_store.graphml", "w") as writer:
        writer.write(lomap_network.to_graphml())

    # Define the Protein and Solvent Components
    protein = ProteinComponent.from_pdb_file(args.protein)
    solvent = SolventComponent(positive_ion='Na', negative_ion='Cl', neutralize=True, ion_concentration=0.15*unit.molar)

    # Extract the relevant edge for the lig_ejm_31 -> lig_ejm_47 transform in the radial graph
    ejm_31_to_ejm_47 = [edge for edge in lomap_network.edges if edge.componentB.name == "lig_ejm_47"][0]

    # Create the four ChemicalSystems
    ejm_31_complex = ChemicalSystem({'ligand': ejm_31_to_ejm_47.componentA, 'solvent': solvent, 'protein': protein}, name=ejm_31_to_ejm_47.componentA.name)
    ejm_31_solvent = ChemicalSystem({'ligand': ejm_31_to_ejm_47.componentA, 'solvent': solvent}, name=ejm_31_to_ejm_47.componentA.name)
    ejm_47_complex = ChemicalSystem({'ligand': ejm_31_to_ejm_47.componentB, 'solvent': solvent, 'protein': protein}, name=ejm_31_to_ejm_47.componentB.name)
    ejm_47_solvent = ChemicalSystem({'ligand': ejm_31_to_ejm_47.componentB, 'solvent': solvent}, name=ejm_31_to_ejm_47.componentB.name)

    # Create the default settings for RBFE Protocol
    rbfe_settings = RelativeHybridTopologyProtocol.default_settings()

    # Create RBFE Protocol class
    rbfe_protocol = RelativeHybridTopologyProtocol(settings=rbfe_settings)

    # Create the transformations
    transformation_complex = Transformation(
        stateA=ejm_31_complex,
        stateB=ejm_47_complex,
        mapping=ejm_31_to_ejm_47,
        protocol=rbfe_protocol,
        name=f"{ejm_31_complex.name}_{ejm_47_complex.name}_complex"
    )
    transformation_solvent = Transformation(
        stateA=ejm_31_solvent,
        stateB=ejm_47_solvent,
        mapping=ejm_31_to_ejm_47,
        protocol=rbfe_protocol,
        name=f"{ejm_31_solvent.name}_{ejm_47_solvent.name}_solvent"
    )

    # Create the DAGs for the transformations
    complex_dag = transformation_complex.create()
    solvent_dag = transformation_solvent.create()

    # Dry-run for complex and solvent transformations
    complex_unit = list(complex_dag.protocol_units)[0]
    complex_unit.run(dry=True, verbose=True)

    solvent_unit = list(solvent_dag.protocol_units)[0]
    solvent_unit.run(dry=True, verbose=True)

    # Create the output directories
    complex_path = pathlib.Path(args.output) / 'complex'
    complex_path.mkdir(parents=True, exist_ok=True)

    solvent_path = pathlib.Path(args.output) / 'solvent'
    solvent_path.mkdir(parents=True, exist_ok=True)

    # Execute the DAGs
    complex_dag_results = execute_DAG(complex_dag, scratch_basedir=complex_path, shared_basedir=complex_path)
    solvent_dag_results = execute_DAG(solvent_dag, scratch_basedir=solvent_path, shared_basedir=solvent_path)

    # Gather the results
    complex_results = rbfe_protocol.gather([complex_dag_results])
    solvent_results = rbfe_protocol.gather([solvent_dag_results])

    # Print the results
    print(f"Complex dG: {complex_results.get_estimate()}, err {complex_results.get_uncertainty()}")
    print(f"Solvent dG: {solvent_results.get_estimate()}, err {solvent_results.get_uncertainty()}")

if __name__ == "__main__":
    main()
