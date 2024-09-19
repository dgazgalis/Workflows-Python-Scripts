import argparse
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from openfe import SmallMoleculeComponent
from openff.toolkit import Molecule
from openfe.setup import LomapAtomMapper
from openfe.setup.ligand_network_planning import (
    generate_minimal_spanning_network,
    generate_lomap_network,
    generate_radial_network
)
from openfe.utils.atommapping_network_plotting import plot_atommapping_network
import networkx as nx
import matplotlib.pyplot as plt

# Load ligands using RDKit and compute 2D coordinates
def load_ligands_rdkit(input_file):
    ligands_rdmol = [mol for mol in Chem.SDMolSupplier(input_file, removeHs=False)]
    for ligand in ligands_rdmol:
        AllChem.Compute2DCoords(ligand)
    return ligands_rdmol

# Visualize ligands using RDKit
def visualize_ligands(ligands_rdmol):
    Chem.Draw.MolsToGridImage(ligands_rdmol)

# Load ligands using OpenFF toolkit
def load_ligands_openff(input_file):
    return Molecule.from_file(input_file)

# Convert OpenFF molecules to OpenFE SmallMoleculeComponents
def convert_to_openfe(ligands_openff):
    return [SmallMoleculeComponent.from_openff(sdf) for sdf in ligands_openff]

# Generate different types of networks
def generate_networks(ligand_mols):
    mapper = LomapAtomMapper()
    mst_network = generate_minimal_spanning_network(
        ligands=ligand_mols,
        scorer=openfe.lomap_scorers.default_lomap_score,
        mappers=[LomapAtomMapper()]
    )
    lomap_network = generate_lomap_network(
        molecules=ligand_mols,
        scorer=openfe.lomap_scorers.default_lomap_score,
        mappers=[LomapAtomMapper()]
    )
    radial_network = generate_radial_network(
        ligands=ligand_mols[1:],
        central_ligand=ligand_mols[0],
        mappers=[LomapAtomMapper()]
    )
    return mst_network, lomap_network, radial_network

# Visualize a network
def visualize_network(network):
    plot_atommapping_network(network)

# Save a network in GraphML format
def save_network(network, filename):
    with open(filename, "w") as writer:
        writer.write(network.to_graphml())

# Convert GraphML to PNG
def convert_graphml_to_png(graphml_file, png_file):
    G = nx.read_graphml(graphml_file)
    pos = nx.spring_layout(G)  # Layout algorithm for positioning nodes
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=10, node_color='lightblue')
    plt.savefig(png_file, format='png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Route planning with Open Free Energy for relative binding free energy calculations.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input SDF file containing ligands.")
    args = parser.parse
    
    ligands_rdmol = load_ligands_rdkit(args.input)
    visualize_ligands(ligands_rdmol)
    ligands_openff = load_ligands_openff(args.input)
    ligand_mols = convert_to_openfe(ligands_openff)
    print("name: ", ligand_mols[0].name)
    mst_network, lomap_network, radial_network = generate_networks(ligand_mols)
    visualize_network(mst_network)
    save_network(mst_network, "mst_network.graphml")
    save_network(lomap_network, "lomap_network.graphml")
    save_network(radial_network, "radial_network.graphml")

    # Convert GraphML files to PNG images
    convert_graphml_to_png("mst_network.graphml", "mst_network.png")
    convert_graphml_to_png("lomap_network.graphml", "lomap_network.png")
    convert_graphml_to_png("radial_network.graphml", "radial_network.png")

if __name__ == "__main__":
    main()