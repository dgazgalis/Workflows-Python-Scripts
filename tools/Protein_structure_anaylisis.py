import os
import numpy as np
import requests
from Bio.PDB import PDBParser, DSSP, Superimposer, PDBList
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import multiprocessing as mp
from Bio.PDB.ResidueDepth import ResidueDepth

def download_pdb_files(pdb_ids, download_dir):
    pdbl = PDBList()
    for pdb_id in pdb_ids:
        pdbl.retrieve_pdb_file(pdb_id, pdir=download_dir, file_format='pdb')

def load_protein_structure(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    return structure

def analyze_secondary_structure(structure, model):
    dssp = DSSP(model, pdb_file)
    secondary_structures = [dssp[key][2] for key in dssp.keys()]
    return secondary_structures

def analyze_hydrogen_bonds(structure, model):
    dssp = DSSP(model, pdb_file)
    hydrogen_bonds = [dssp[key][6] for key in dssp.keys()]  # Accessible surface area
    return hydrogen_bonds

def calculate_rmsd(structure1, structure2):
    sup = Superimposer()
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_name() == 'CA']
    sup.set_atoms(atoms1, atoms2)
    return sup.rms

def calculate_distance_matrix(structures):
    n = len(structures)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            rmsd = calculate_rmsd(structures[i], structures[j])
            distance_matrix[i, j] = rmsd
            distance_matrix[j, i] = rmsd
    return distance_matrix

def find_most_dissimilar_structures(distance_matrix):
    max_distance = np.max(distance_matrix)
    max_indices = np.where(distance_matrix == max_distance)
    return max_indices[0]

def cluster_structures(distance_matrix, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(distance_matrix)
    return labels

def calculate_sasa(structure):
    model = structure[0]
    rd = ResidueDepth(model)
    sasa_values = rd.get_sasa()
    return sasa_values

def find_common_residue_numbers(structures):
    residue_numbers = [set(res.get_id()[1] for res in structure.get_residues()) for structure in structures]
    common_residue_numbers = set.intersection(*residue_numbers)
    return sorted(common_residue_numbers)

def truncate_structures(structures, common_residue_numbers):
    truncated_structures = []
    for structure in structures:
        model = structure[0]
        truncated_model = model.copy()
        for chain in truncated_model:
            chain.child_list = [res for res in chain if res.get_id()[1] in common_residue_numbers]
        truncated_structures.append(truncated_model)
    return truncated_structures

def plot_results(secondary_structures_list, hydrogen_bonds_list, sasa_list, pdb_ids, labels):
    plt.figure(figsize=(15, 10))

    for i, (secondary_structures, hydrogen_bonds, sasa_values) in enumerate(zip(secondary_structures_list, hydrogen_bonds_list, sasa_list)):
        plt.subplot(3, len(pdb_ids), i + 1)
        plt.hist(secondary_structures, bins=3, edgecolor='black')
        plt.title(f'Secondary Structure Distribution - {pdb_ids[i]} (Cluster {labels[i]})')
        plt.xticks([0, 1, 2], ['C', 'H', 'E'])

        plt.subplot(3, len(pdb_ids), len(pdb_ids) + i + 1)
        plt.hist(hydrogen_bonds, bins=50, edgecolor='black')
        plt.title(f'Hydrogen Bonds Distribution - {pdb_ids[i]} (Cluster {labels[i]})')

        plt.subplot(3, len(pdb_ids), 2 * len(pdb_ids) + i + 1)
        plt.hist(sasa_values, bins=50, edgecolor='black')
        plt.title(f'Solvent-Accessible Surface Area (SASA) Distribution - {pdb_ids[i]} (Cluster {labels[i]})')

    plt.tight_layout()
    plt.show()

def main(pdb_ids, download_dir):
    # Download PDB files
    download_pdb_files(pdb_ids, download_dir)
    
    # Initialize lists to store structures and analysis results
    structures = []
    secondary_structures_list = []
    hydrogen_bonds_list = []
    sasa_list = []

    # Load and analyze each protein structure
    for pdb_id in pdb_ids:
        pdb_file = os.path.join(download_dir, f'pdb{pdb_id.lower()}.ent')
        structure = load_protein_structure(pdb_file)
        structures.append(structure)
        
        # Analyze secondary structure and hydrogen bonds
        model = structure[0]
        secondary_structures = analyze_secondary_structure(structure, model)
        hydrogen_bonds = analyze_hydrogen_bonds(structure, model)
        
        # Calculate SASA
        sasa_values = calculate_sasa(structure)
        
        # Store analysis results
        secondary_structures_list.append(secondary_structures)
        hydrogen_bonds_list.append(hydrogen_bonds)
        sasa_list.append(sasa_values)
    
    # Find common residue numbers
    common_residue_numbers = find_common_residue_numbers(structures)
    
    # Truncate structures to have the same number of residues
    truncated_structures = truncate_structures(structures, common_residue_numbers)
    
    # Calculate distance matrix based on RMSD
    distance_matrix = calculate_distance_matrix(truncated_structures)
    
    # Find the most dissimilar structures
    dissimilar_indices = find_most_dissimilar_structures(distance_matrix)
    print(f'Most dissimilar structures are: {pdb_ids[dissimilar_indices[0]]} and {pdb_ids[dissimilar_indices[1]]}')

    # Cluster structures
    labels = cluster_structures(distance_matrix)

    # Plot the results
    plot_results(secondary_structures_list, hydrogen_bonds_list, sasa_list, pdb_ids, labels)

if __name__ == "__main__":
    pdb_ids = ['1A2B', '1C2D', '1E2F']  # Replace with your list of PDB IDs
    download_dir = 'path_to_download_directory'  # Replace with your download directory
    main(pdb_ids, download_dir)

