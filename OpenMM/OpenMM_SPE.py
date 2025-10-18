# OpenMM Script for Single-Point Energy Calculation
#
# This script demonstrates how to set up a molecular system and
# calculate its potential energy without running a simulation.
#
# The workflow is:
# 1. Load Input Files
# 2. Choose a Force Field
# 3. Create the System
# 4. Create a Simulation Context
# 5. Calculate and Print the Energy

# --- 0. Import Necessary Libraries ---
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

try:
    # --- 1. Load Input Files ---
    # Load the PDB file containing the molecular structure and atom positions.
    print("Loading PDB file...")
    pdb = PDBFile('input.pdb')

    # --- 2. Choose a Force Field ---
    # Load the force field to define the physics of the system.
    print("Loading force field...")
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # --- 3. Create the System ---
    # Solvate the molecule in a water box and create the OpenMM System object.
    print("Creating system and adding solvent...")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, padding=1.0*nanometers)
    
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometers, constraints=HBonds, rigidWater=True)

    # --- 4. Create a Simulation Context ---
    # To calculate energy, we need a 'Context'. This is created by the
    # Simulation object. An integrator is required, but since we are not
    # running dynamics, we can provide a simple dummy integrator.
    print("Creating simulation context...")
    dummy_integrator = VerletIntegrator(0.001*picoseconds)
    simulation = Simulation(modeller.topology, system, dummy_integrator)
    simulation.context.setPositions(modeller.positions)

    # --- 5. Calculate and Print the Energy ---
    # Get the current state of the system from the context, requesting energy.
    # Then, retrieve the potential energy from that state.
    print("Calculating potential energy...")
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    
    print("\n" + "="*40)
    print(f"System Potential Energy: {potential_energy}")
    print("="*40)


except Exception as e:
    print(f"An error occurred: {e}")
    print("\nThis script requires an 'input.pdb' file in the same directory.")
    print("You can get a standard alanine dipeptide PDB here:")
    print("https://raw.githubusercontent.com/openmm/openmm/master/examples/input.pdb")

