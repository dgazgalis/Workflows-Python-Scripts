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
from pyscf.geomopt.geometric_solver import optimize
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

# Utility functions for calculations and constraints
def create_constraints_file(optimize_h_only, mol):
    constraints = []
    if optimize_h_only:
        constraints.append("$freeze\n")
        non_h_indices = [i+1 for i, atom in enumerate(mol._atom) if atom[0] != 'H']
        constraints.append(f"xyz {','.join(map(str, non_h_indices))}\n")
        constraints.append("$end\n")
    
    # Join the constraints into a single string
    constraints_str = ''.join(constraints)
    
    # Write constraints to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(constraints_str)
        return temp_file.name

# Core optimization functions
def optimize_dft(mol, args, optimize_h_only=False):
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

    gradients = []
    def callback(envs):
        gradients.append(envs['gradients'])

    start_time = time.time()
    
    constraints_file = create_constraints_file(optimize_h_only, mol)
    params = {"constraints": constraints_file}
    mol_eq = optimize(mf_GPU, maxsteps=500, callback=callback, **params)

    print('Geometry optimization took', time.time() - start_time, 's')

    final_energy = mf_GPU.kernel(mol_eq, dm0=dm0)
    print('Final energy:', final_energy)

    print("\nGradient norm at each step:")
    for i, grad in enumerate(gradients):
        print(f"Step {i+1}: {np.linalg.norm(grad):.6f}")

    return mol_eq

# Output function
def save_optimized_geometry(mol_eq, base_filename, method):
    optimized_xyz = f"{base_filename}_optimized_{method}.xyz"
    with open(optimized_xyz, 'w') as f:
        f.write(f"{mol_eq.natm}\n\n")
        for i, coord in enumerate(mol_eq.atom_coords()):
            symbol = mol_eq.atom_symbol(i)
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    print(f"Optimized geometry saved to {optimized_xyz}")

    optimized_rdkit_mol = pyscf_mol_to_rdkit(mol_eq)
    optimized_sdf = f"{base_filename}_optimized_{method}.sdf"
    writer = Chem.SDWriter(optimized_sdf)
    writer.write(optimized_rdkit_mol)
    writer.close()
    print(f"Optimized geometry saved to {optimized_sdf}")

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
    print(f"Performing DFT optimization (charge: {args.charge})...")
    mol_eq = optimize_dft(mol, args, optimize_h_only=args.optimize_h)
    save_optimized_geometry(mol_eq, base_filename, "dft")
        
    sys.exit(0)

# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize molecule geometry using MINDO3 or DFT.")
    parser.add_argument("-i", "--input", required=True, help="Input SDF file")
    parser.add_argument("-c", "--charge", type=int, default=0, help="Molecular charge")
    parser.add_argument("-s", "--spin", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("-b", "--basis", default="def2-svp", help="Basis set for DFT")
    parser.add_argument("-f", "--functional", default="b3lyp", help="DFT functional")
    parser.add_argument("-n", "--cores", type=int, help="Number of CPU cores to use")
    parser.add_argument("--optimize-h", action="store_true", help="Optimize only hydrogen atoms")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"An unhandled error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)