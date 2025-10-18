#!/usr/bin/env python3
"""
Aligned Fragment Analyzer for AlphaFold 3 Predictions

This analyzer first aligns all protein structures using CA atoms, then analyzes
fragment distributions in the aligned coordinate system. This is essential for
meaningful probability distributions since raw AF3 predictions have different
orientations and positions.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from Bio.PDB import *
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import argparse

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class AlignedFragmentAnalyzer:
    """Analyzes fragment distributions after aligning protein structures."""

    def __init__(self, dataset_path: str, grid_resolution: float = 1.0):
        """
        Initialize the analyzer.

        Args:
            dataset_path: Path to dataset directory
            grid_resolution: Grid spacing in Angstroms
        """
        self.dataset_path = dataset_path
        self.grid_resolution = grid_resolution

        # Storage for aligned structures and fragments
        self.reference_structure = None
        self.aligned_structures = []
        self.fragment_coordinates = []
        self.prediction_count = 0

        # Grid data
        self.grid_bounds = None
        self.grid_data = None
        self.grid_dimensions = None

    def get_ca_atoms(self, structure):
        """Extract CA atoms from protein chain A for alignment."""
        ca_atoms = []
        for model in structure:
            for chain in model:
                if chain.id == 'A':  # Protein chain
                    for residue in chain:
                        if residue.has_id('CA'):
                            ca_atoms.append(residue['CA'])
        return ca_atoms

    def read_cif_file(self, filepath: str):
        """Read a CIF file and return the structure."""
        parser = MMCIFParser(QUIET=True)
        try:
            structure = parser.get_structure("model", filepath)
            return structure
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            return None

    def align_structure_to_reference(self, mobile_structure):
        """Align mobile structure to reference structure using CA atoms."""
        if self.reference_structure is None:
            raise ValueError("No reference structure set")

        ref_ca = self.get_ca_atoms(self.reference_structure)
        mob_ca = self.get_ca_atoms(mobile_structure)

        # Ensure both structures have the same number of CA atoms
        min_len = min(len(ref_ca), len(mob_ca))
        if min_len == 0:
            print("Warning: No CA atoms found for alignment")
            return mobile_structure

        ref_ca = ref_ca[:min_len]
        mob_ca = mob_ca[:min_len]

        # Perform superimposition
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_ca, mob_ca)

        # Collect ALL atoms from the mobile structure (protein + ligands)
        all_atoms = []
        for model in mobile_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        all_atoms.append(atom)

        # Apply the same transformation to ALL atoms
        super_imposer.apply(all_atoms)

        return mobile_structure

    def find_prediction_directories(self) -> List[str]:
        """Find all seed-X_sample-Y directories in the dataset."""
        directories = []

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        for item in os.listdir(self.dataset_path):
            if item.startswith('seed-') and '_sample-' in item:
                cif_path = os.path.join(self.dataset_path, item, 'model.cif')
                if os.path.exists(cif_path):
                    directories.append(os.path.join(self.dataset_path, item))

        return sorted(directories)

    def align_all_structures(self):
        """Align all structures in the dataset to a common reference frame."""
        print("Aligning all structures to common reference frame...")

        directories = self.find_prediction_directories()
        print(f"Found {len(directories)} prediction directories")

        self.aligned_structures = []
        self.reference_structure = None

        for i, pred_dir in enumerate(directories):
            cif_path = os.path.join(pred_dir, 'model.cif')

            if i % 20 == 0:  # Progress update
                print(f"Processing {i+1}/{len(directories)}: {os.path.basename(pred_dir)}")

            structure = self.read_cif_file(cif_path)
            if structure is None:
                continue

            # Use first structure as reference
            if self.reference_structure is None:
                self.reference_structure = structure
                structure.id = f"reference_{os.path.basename(pred_dir)}"
                self.aligned_structures.append(structure)
                print(f"Using {os.path.basename(pred_dir)} as reference structure")
            else:
                # Align to reference
                aligned_structure = self.align_structure_to_reference(structure)
                aligned_structure.id = f"aligned_{os.path.basename(pred_dir)}"
                self.aligned_structures.append(aligned_structure)

        print(f"Successfully aligned {len(self.aligned_structures)} structures")

    def extract_fragments_from_aligned_structures(self):
        """Extract fragment coordinates from all aligned structures."""
        print("Extracting fragment coordinates from aligned structures...")

        self.fragment_coordinates = []
        self.prediction_count = 0

        for i, structure in enumerate(self.aligned_structures):
            if i % 20 == 0:
                print(f"Extracting from structure {i+1}/{len(self.aligned_structures)}")

            structure_fragments = []

            for model in structure:
                for chain in model:
                    # Skip protein chain A
                    if chain.id == 'A':
                        continue

                    # Look for fragment/ligand chains
                    chain_coords = []
                    for residue in chain:
                        # Check if this is a ligand/fragment residue
                        res_name = residue.get_resname()
                        if (res_name.startswith('LIG_') or
                            res_name in ['UNL', 'HET', 'LIG'] or
                            len(res_name) <= 3):  # Catch various ligand naming conventions

                            for atom in residue:
                                chain_coords.append(atom.get_coord())

                    if chain_coords:
                        structure_fragments.append(np.array(chain_coords))

            if structure_fragments:
                self.fragment_coordinates.extend(structure_fragments)
                self.prediction_count += 1

        print(f"Extracted {len(self.fragment_coordinates)} fragment instances from {self.prediction_count} predictions")

    def create_aligned_probability_grid(self, sigma: float = 2.0):
        """Create probability grid from aligned fragment coordinates."""
        if not self.fragment_coordinates:
            raise ValueError("No fragment coordinates available. Run alignment and extraction first.")

        print("Creating probability grid from aligned coordinates...")

        # Calculate grid bounds
        all_coords = np.concatenate(self.fragment_coordinates, axis=0)
        padding = 5.0
        min_bounds = np.min(all_coords, axis=0) - padding
        max_bounds = np.max(all_coords, axis=0) + padding

        self.grid_bounds = (min_bounds, max_bounds)

        # Calculate grid dimensions
        grid_size = max_bounds - min_bounds
        self.grid_dimensions = np.ceil(grid_size / self.grid_resolution).astype(int)

        print(f"Aligned grid bounds: {min_bounds} to {max_bounds}")
        print(f"Aligned grid dimensions: {self.grid_dimensions}")
        print(f"Grid resolution: {self.grid_resolution} Å")

        # Initialize grid
        self.grid_data = np.zeros(self.grid_dimensions, dtype=np.float32)

        # Create coordinate grids
        x = np.linspace(min_bounds[0], max_bounds[0], self.grid_dimensions[0])
        y = np.linspace(min_bounds[1], max_bounds[1], self.grid_dimensions[1])
        z = np.linspace(min_bounds[2], max_bounds[2], self.grid_dimensions[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X, Y, Z], axis=-1)

        # Add Gaussian density for each fragment
        total_fragments = len(self.fragment_coordinates)

        for frag_idx, fragment_coords in enumerate(self.fragment_coordinates):
            if frag_idx % 50 == 0:
                print(f"Processing fragment {frag_idx+1}/{total_fragments}")

            # Calculate center of mass for this fragment
            fragment_center = np.mean(fragment_coords, axis=0)

            # Add Gaussian density centered at fragment center
            distances_sq = np.sum((grid_points - fragment_center)**2, axis=-1)
            density = np.exp(-distances_sq / (2 * sigma**2))

            self.grid_data += density

        # Normalize by total number of predictions
        self.grid_data /= self.prediction_count

        print(f"Aligned grid statistics:")
        print(f"  Max probability: {np.max(self.grid_data):.6f}")
        print(f"  Mean probability: {np.mean(self.grid_data):.6f}")
        print(f"  Non-zero voxels: {np.sum(self.grid_data > 0)}")

    def write_vtk_file(self, output_path: str):
        """Write the aligned probability grid to a VTK file."""
        if self.grid_data is None:
            raise ValueError("No grid data available. Run create_aligned_probability_grid() first.")

        print(f"Writing aligned VTK file: {output_path}")

        with open(output_path, 'w') as f:
            # VTK header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Aligned AlphaFold 3 Fragment Probability Distribution\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")

            # Grid dimensions and spacing
            f.write(f"DIMENSIONS {self.grid_dimensions[0]} {self.grid_dimensions[1]} {self.grid_dimensions[2]}\n")
            f.write(f"ORIGIN {self.grid_bounds[0][0]} {self.grid_bounds[0][1]} {self.grid_bounds[0][2]}\n")
            f.write(f"SPACING {self.grid_resolution} {self.grid_resolution} {self.grid_resolution}\n")

            # Point data
            total_points = np.prod(self.grid_dimensions)
            f.write(f"POINT_DATA {total_points}\n")
            f.write("SCALARS probability float 1\n")
            f.write("LOOKUP_TABLE default\n")

            # Write probability data
            flat_data = self.grid_data.flatten(order='F')  # Fortran order for VTK
            for value in flat_data:
                f.write(f"{value:.6f}\n")

        print(f"Aligned VTK file written successfully")

    def save_aligned_structures_pdb(self, output_path: str):
        """Save all aligned structures to a single PDB file for verification."""
        print(f"Saving aligned structures to PDB: {output_path}")

        # Create a new structure to hold all models
        combined_structure = Structure.Structure("aligned_combined")

        # Add each aligned structure as a separate model
        for i, struct in enumerate(self.aligned_structures):
            model = struct[0]  # Get the first (and typically only) model
            model.id = i  # Assign unique model ID
            combined_structure.add(model)

        # Write to PDB
        io = PDBIO()
        io.set_structure(combined_structure)

        class AllAtomSelect(Select):
            def accept_atom(self, atom):
                return True
            def accept_residue(self, residue):
                return True

        io.save(output_path, AllAtomSelect())
        print(f"Saved {len(self.aligned_structures)} aligned structures to {output_path}")

    def generate_alignment_report(self, output_path: str):
        """Generate a report on the alignment process."""
        with open(output_path, 'w') as f:
            f.write("Aligned AlphaFold 3 Fragment Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total structures aligned: {len(self.aligned_structures)}\n")
            f.write(f"Reference structure: {self.reference_structure.id if self.reference_structure else 'None'}\n")
            f.write(f"Total fragment instances: {len(self.fragment_coordinates)}\n")
            f.write(f"Predictions with fragments: {self.prediction_count}\n\n")

            if self.grid_data is not None:
                f.write("Aligned Grid Information:\n")
                f.write(f"  Resolution: {self.grid_resolution} Å\n")
                f.write(f"  Dimensions: {self.grid_dimensions}\n")
                f.write(f"  Bounds: {self.grid_bounds[0]} to {self.grid_bounds[1]}\n")
                f.write(f"  Max probability: {np.max(self.grid_data):.6f}\n")
                f.write(f"  Mean probability: {np.mean(self.grid_data):.6f}\n")
                f.write(f"  Non-zero voxels: {np.sum(self.grid_data > 0)}\n\n")

            # Add alignment quality metrics if available
            if self.reference_structure and len(self.aligned_structures) > 1:
                f.write("Alignment Quality:\n")
                f.write("  All structures aligned using CA atoms from protein chain A\n")
                f.write("  Transformation applied to all atoms (protein + fragments)\n")
                f.write("  Fragments analyzed in common reference frame\n")

        print(f"Alignment report written to: {output_path}")


def main():
    """Main function to run the aligned fragment analysis."""
    parser = argparse.ArgumentParser(description='Aligned AlphaFold 3 fragment analysis')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--output', '-o', default='aligned_fragment_analysis',
                        help='Output filename prefix (default: aligned_fragment_analysis)')
    parser.add_argument('--resolution', '-r', type=float, default=1.5,
                        help='Grid resolution in Angstroms (default: 1.5)')
    parser.add_argument('--sigma', '-s', type=float, default=2.5,
                        help='Gaussian kernel sigma (default: 2.5)')
    parser.add_argument('--save_structures', action='store_true',
                        help='Save aligned structures to PDB file for verification')

    args = parser.parse_args()

    print("Aligned AlphaFold 3 Fragment Analyzer")
    print("=" * 40)

    # Initialize analyzer
    analyzer = AlignedFragmentAnalyzer(
        dataset_path=args.dataset_path,
        grid_resolution=args.resolution
    )

    try:
        # Step 1: Align all structures
        analyzer.align_all_structures()

        if len(analyzer.aligned_structures) == 0:
            print("ERROR: No structures could be aligned!")
            return 1

        # Step 2: Extract fragment coordinates from aligned structures
        analyzer.extract_fragments_from_aligned_structures()

        if len(analyzer.fragment_coordinates) == 0:
            print("ERROR: No fragments found in aligned structures!")
            return 1

        # Step 3: Create probability density grid
        analyzer.create_aligned_probability_grid(sigma=args.sigma)

        # Step 4: Write outputs
        vtk_path = f"{args.output}.vtk"
        report_path = f"{args.output}_report.txt"

        analyzer.write_vtk_file(vtk_path)
        analyzer.generate_alignment_report(report_path)

        # Optional: Save aligned structures for verification
        if args.save_structures:
            pdb_path = f"{args.output}_aligned_structures.pdb"
            analyzer.save_aligned_structures_pdb(pdb_path)

        print(f"\nAligned analysis complete!")
        print(f"VTK file: {vtk_path}")
        print(f"Report: {report_path}")
        if args.save_structures:
            print(f"Aligned structures PDB: {pdb_path}")

        print(f"\nFor ParaView visualization:")
        print(f"1. Open {vtk_path} in ParaView")
        print(f"2. Set representation to 'Volume'")
        print(f"3. Adjust opacity transfer function")
        print(f"4. Fragment hotspots should now be properly localized!")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())