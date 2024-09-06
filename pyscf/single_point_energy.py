import argparse
import os
import time
import multiprocessing
import sys
import traceback

import numpy as np
from rdkit import Chem

import pyscf
from pyscf import dft, gto, lib
from pyscf import scf as cpu_scf
from gpu4pyscf.dft import rks as gpu_rks

# Logging and back tracing 
import logging
logging.basicConfig(level=logging.DEBUG)

# Wrap the main execution in a try-except block
def run_with_error_handling(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            sys.exit(1)
    return wrapper

# File conversion and molecule representation functions
def sdf_to_xyz(sdf_file, xyz_file):
    mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    
    with open(xyz_file, 'w') as f:
        f.write(f"{len(symbols)}\n\n")
        for symbol, (x, y, z) in zip(symbols, coords):
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    
    return mol

def pyscf_mol_to_rdkit(pyscf_mol):
    rdkit_mol = Chem.RWMol()
    for i in range(pyscf_mol.natm):
        atom = Chem.Atom(pyscf_mol.atom_symbol(i))
        rdkit_mol.AddAtom(atom)
    
    conf = Chem.Conformer(pyscf_mol.natm)
    for i, coord in enumerate(pyscf_mol.atom_coords()):
        conf.SetAtomPosition(i, coord)
    rdkit_mol.AddConformer(conf)
    
    return rdkit_mol

# Core energy calculation function
def calculate_energy(mol, args):
    try:
        # Initialize the GPU version of RKS
        mf_GPU = gpu_rks.RKS(mol, xc=args.functional)
        mf_GPU = mf_GPU.density_fit()
        print("Using GPU acceleration")
        
        # Create initial guess using CPU version
        mf_CPU = dft.RKS(mol, xc=args.functional)
        dm0 = mf_CPU.get_init_guess(key='minao')
        
        # Convert to GPU array
        import cupy
        dm0 = cupy.asarray(dm0)
        
    except Exception as e:
        print(f"GPU acceleration failed: {e}")
        print(f"Falling back to CPU with {args.cores} cores")
        mf_GPU = dft.RKS(mol, xc=args.functional)
        mf_GPU = mf_GPU.density_fit()
        dm0 = mf_GPU.get_init_guess(key='minao')

    mf_GPU.disp = 'd3bj'
    mf_GPU.grids.level = 3
    mf_GPU.conv_tol = 1e-10
    mf_GPU.max_cycle = 500
    
    start_time = time.time()
    final_energy = mf_GPU.kernel(mol, dm0=dm0)
    print('Energy calculation took', time.time() - start_time, 's')
    print('Final energy:', final_energy)

    return final_energy, mol

# Output function
def save_energy_and_geometry(final_energy, mol, base_filename, method):
    energy_file = f"{base_filename}_energy_{method}.txt"
    with open(energy_file, 'w') as f:
        f.write(f"Final Energy: {final_energy:.8f} Hartree\n")
    print(f"Final energy saved to {energy_file}")

    optimized_xyz = f"{base_filename}_geometry_{method}.xyz"
    with open(optimized_xyz, 'w') as f:
        f.write(f"{mol.natm}\n\n")
        for i, coord in enumerate(mol.atom_coords()):
            symbol = mol.atom_symbol(i)
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    print(f"Geometry saved to {optimized_xyz}")

    optimized_rdkit_mol = pyscf_mol_to_rdkit(mol)
    optimized_sdf = f"{base_filename}_geometry_{method}.sdf"
    writer = Chem.SDWriter(optimized_sdf)
    writer.write(optimized_rdkit_mol)
    writer.close()
    print(f"Geometry saved to {optimized_sdf}")

# Main execution function
@run_with_error_handling
def main(args):
    xyz_file = os.path.splitext(args.input)[0] + ".xyz"
    rdkit_mol = sdf_to_xyz(args.input, xyz_file)

    if args.cores is None:
        args.cores = multiprocessing.cpu_count()
    lib.num_threads(args.cores)
    print(f"Using {args.cores} CPU cores")

    mol = gto.M(
        atom=xyz_file,
        charge=args.charge,
        spin=args.spin - 1,
        basis=args.basis
    )

    base_filename = os.path.splitext(args.input)[0]
    print(f"Performing DFT energy calculation (charge: {args.charge})...")
    final_energy, mol = calculate_energy(mol, args)
    save_energy_and_geometry(final_energy, mol, base_filename, "dft")
        
    sys.exit(0)

# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate molecule energy using DFT.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-c", "--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument("-s", "--spin", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("-b", "--basis", default="def2-svp", help="Basis set for DFT")
    parser.add_argument("-f", "--functional", default="b3lyp", help="DFT functional")
    parser.add_argument("-n", "--cores", type=int, help="Number of CPU cores to use")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"An unhandled error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)