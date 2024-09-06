import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial import cKDTree
import pymol
from pymol import cmd
import argparse

def load_complex(pdb_file):
    pdb = PandasPdb().read_pdb(pdb_file)
    protein = pdb.df['ATOM']
    ligand = pdb.df['HETATM']
    return protein, ligand

def calculate_center_of_mass(atoms):
    coords = atoms[['x_coord', 'y_coord', 'z_coord']].values
    masses = atoms['element_symbol'].map({'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'S': 32.065}).values
    return np.average(coords, axis=0, weights=masses)

def find_closest_surface_atom(protein, ligand_com):
    protein_coords = protein[['x_coord', 'y_coord', 'z_coord']].values
    tree = cKDTree(protein_coords)
    distances, indices = tree.query(ligand_com, k=100)
    min_neighbors = float('inf')
    surface_atom_index = None
    for idx in indices:
        neighbors = tree.query_ball_point(protein_coords[idx], r=5)
        if len(neighbors) < min_neighbors:
            min_neighbors = len(neighbors)
            surface_atom_index = idx
    return protein_coords[surface_atom_index]

def calculate_displacement_vector(surface_atom, ligand_com):
    return ligand_com - surface_atom

def normalize_and_scale_vector(vector, scale=15):
    return scale * vector / np.linalg.norm(vector)

def apply_displacement(ligand, displacement_vector):
    ligand['x_coord'] += displacement_vector[0]
    ligand['y_coord'] += displacement_vector[1]
    ligand['z_coord'] += displacement_vector[2]
    return ligand

def calculate_nearest_atom_distance(protein, ligand):
    protein_coords = protein[['x_coord', 'y_coord', 'z_coord']].values
    ligand_coords = ligand[['x_coord', 'y_coord', 'z_coord']].values
    protein_tree = cKDTree(protein_coords)
    ligand_tree = cKDTree(ligand_coords)
    min_distance, _ = protein_tree.query(ligand_coords, k=1)
    return np.min(min_distance)

def write_displaced_complex(protein, ligand, output_file):
    ppdb = PandasPdb()
    ppdb.df['ATOM'] = protein
    ppdb.df['HETATM'] = ligand
    ppdb.to_pdb(output_file)
    print(f"Displaced complex written to {output_file}")

def visualize_complex(protein, ligand, surface_atom, displacement_vector):
    # Create a temporary PDB file for the displaced complex
    ppdb = PandasPdb()
    ppdb.df['ATOM'] = protein
    ppdb.df['HETATM'] = ligand
    ppdb.to_pdb("displaced_complex_temp.pdb")

    # Initialize PyMOL
    pymol.finish_launching()

    # Load the displaced complex
    cmd.load("displaced_complex_temp.pdb", "complex")

    # Color the protein and ligand
    cmd.color("cyan", "complex and polymer")
    cmd.color("magenta", "complex and organic")
    
    # Show protein as cartoon and ligand as sticks
    cmd.show("cartoon", "complex and polymer")
    cmd.show("sticks", "complex and organic")
    # Highlight the surface atom
    cmd.select("surface_atom", f"complex and id {surface_atom[0]+1}")
    cmd.show("spheres", "surface_atom")
    cmd.color("yellow", "surface_atom")

    # Draw the displacement vector
    start = surface_atom
    end = start + displacement_vector
    cmd.pseudoatom("start", pos=start)
    cmd.pseudoatom("end", pos=end)
    cmd.distance("vector", "start", "end")
    cmd.hide("labels", "vector")
    cmd.color("red", "vector")

    # Center the view
    cmd.zoom("complex")

    # Save the session and image
    cmd.save("displaced_complex.pse")
    cmd.png("displaced_complex.png", width=1000, height=1000, dpi=300, ray=1)

    print("Visualization saved as 'displaced_complex.pse' and 'displaced_complex.png'")

def main():
    parser = argparse.ArgumentParser(description="Displace a ligand from a protein-ligand complex.")
    parser.add_argument("-i", "--input_pdb", help="Input PDB file of the protein-ligand complex")
    parser.add_argument("-d", "--distance", type=float, default=15.0, help="Desired displacement distance in Angstroms (default: 15.0)")
    parser.add_argument("-o", "--output", default="displaced_complex.pdb", help="Output PDB file name (default: displaced_complex.pdb)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization using PyMOL")
    args = parser.parse_args()
    
    # Load the protein-ligand complex
    protein, ligand = load_complex(args.input_pdb)
    
    # Use the requested distance from command-line argument
    requested_distance = args.distance
    
    # Calculate center of mass for the ligand
    initial_ligand_com = calculate_center_of_mass(ligand)
    
    # Find the closest surface atom on the protein
    surface_atom = find_closest_surface_atom(protein, initial_ligand_com)
    
    # Calculate initial displacement vector
    initial_vector = calculate_displacement_vector(surface_atom, initial_ligand_com)
    
    # Normalize and scale the vector
    displacement_vector = normalize_and_scale_vector(initial_vector, scale=requested_distance)
    
    # Apply displacement to ligand
    displaced_ligand = apply_displacement(ligand, displacement_vector)
    
    # Calculate final displacement vector
    final_ligand_com = calculate_center_of_mass(displaced_ligand)
    final_vector = calculate_displacement_vector(surface_atom, final_ligand_com)
    
    # Calculate nearest atom distance after displacement
    nearest_atom_distance = calculate_nearest_atom_distance(protein, displaced_ligand)
    
    # Check if the produced displacement matches the requested distance
    actual_displacement = np.linalg.norm(final_vector)
    displacement_difference = abs(actual_displacement - requested_distance)
    
    # Report results
    print(f"Requested displacement distance: {requested_distance} Angstroms")
    print(f"Actual displacement distance: {actual_displacement:.4f} Angstroms")
    print(f"Difference: {displacement_difference:.4f} Angstroms")
    print(f"Final displacement vector: {final_vector}")
    print(f"Coordinates of surface atom used: {surface_atom}")
    print(f"Nearest atom distance after displacement: {nearest_atom_distance:.4f} Angstroms")
    
    if displacement_difference < 0.01:  # Allow for small floating-point errors
        print("The produced displacement matches the requested distance.")
    else:
        print("Warning: The produced displacement does not match the requested distance.")
        
    # Write the displaced complex to a PDB file
    write_displaced_complex(protein, displaced_ligand, args.output)

    # Visualize the complex if the --visualize flag is set
    if args.visualize:
        visualize_complex(protein, displaced_ligand, surface_atom, final_vector)

if __name__ == "__main__":
    main()