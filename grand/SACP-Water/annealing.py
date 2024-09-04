import argparse
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from openmmtools.integrators import BAOABIntegrator
from sys import stdout
import numpy as np
import mdtraj
import grand
from tqdm import tqdm
import time

def main(adams_value, mode, previous_pdb=None):
    # Determine which PDB file to use based on the Adams value
    if adams_value == 15:
        pdb_file = 'uvt_inital.pdb'
    else:
        pdb_file = previous_pdb
  
    print(f"Using PDB file: {pdb_file}")
    # Read the total solvent count from solvent_count.log
    with open('solvent_count.log', 'r') as f:
        total_solvent_count = int(f.read().strip())

    # Load the PDB file and count current water molecules
    pdb = PDBFile(pdb_file)
    current_water_count = sum([1 for residue in pdb.topology.residues() if residue.name == 'HOH'])

    # Calculate the number of ghosts to add
    ghosts_to_add = total_solvent_count - current_water_count
    print(f"Total solvent count from log: {total_solvent_count}")
    print(f"Current water molecules: {current_water_count}")
    print(f"Number of ghosts to add: {ghosts_to_add}")

    # Add the calculated number of ghost waters
    pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology, pdb.positions,
                                                                 n=ghosts_to_add, pdb='ghosts_continued.pdb')
    print(f"Added {ghosts_to_add} ghost waters.")

    # Create system
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12*angstrom,
                                     switchDistance=10*angstrom, constraints=HBonds)

    # Fix protein atoms and set their masses to 0
    for i, atom in enumerate(pdb.topology.atoms()):
        if atom.residue.name != 'HOH':
            system.setParticleMass(i, 0 * dalton)

    # Make sure the LJ interactions are being switched
    for f in range(system.getNumForces()):
        force = system.getForce(f)
        if 'NonbondedForce' == force.__class__.__name__:
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(1.0*nanometer)

    # Get the box vectors from the topology
    box_vectors = np.array(pdb.topology.getPeriodicBoxVectors())

    # Define GCMC Sampler using StandardGCMCSystemSampler
    gcmc_mover = grand.samplers.StandardGCMCSystemSampler(
        system=system,
        topology=pdb.topology,
        temperature=298*kelvin,
        adams=adams_value,
        boxVectors=box_vectors,
        ghostFile=f'ghosts_{adams_value}.txt',
        log=f'gcmc_{adams_value}.log',
        overwrite=True
    )

    # Define integrator
    integrator = BAOABIntegrator(298*kelvin, 1.0/picosecond, 0.002*picosecond)

    # Define platform and set precision
    platform = Platform.getPlatformByName('CUDA')
    platform.setPropertyDefaultValue('Precision', 'mixed')

    # Create simulation object
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Set positions, velocities and box vectors
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(298*kelvin)
    simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Prepare the GCMC sampler
    gcmc_mover.initialise(simulation.context, ghosts)

    if mode == 'coarse':
        # Run GCMC/MD equilibration without saving intermediate configurations
        total_steps = 10000
        with tqdm(total=total_steps, desc="Simulation progress", unit="steps") as pbar:
            start_time = time.time()
            for i in range(total_steps):
                simulation.step(500)
                gcmc_mover.move(simulation.context, 50)
                gcmc_mover.report(simulation)
                pbar.update(1)
                
                if i % 100 == 0:  # Update ETA every 100 steps
                    elapsed_time = time.time() - start_time
                    steps_per_second = (i + 1) / elapsed_time
                    eta = (total_steps - i - 1) / steps_per_second
                    pbar.set_postfix({"ETA": f"{eta:.2f}s"})
            
    else:  # fine mode
        # Run GCMC/MD equilibration (100k GCMC moves over 1 ps - 1000 moves every 10 fs)
        num_configurations = 100
        total_steps = 10000
        steps_per_configuration = total_steps // num_configurations
        
        # Run half of the trajectory without saving configurations
        with tqdm(total=total_steps // 2, desc="First half progress", unit="steps") as pbar:
            start_time = time.time()
            for i in range(total_steps // 2):
                simulation.step(500)
                gcmc_mover.move(simulation.context, 50)
                gcmc_mover.report(simulation)
                pbar.update(1)
                
                if i % 100 == 0:  # Update ETA every 100 steps
                    elapsed_time = time.time() - start_time
                    steps_per_second = (i + 1) / elapsed_time
                    eta = (total_steps // 2 - i - 1) / steps_per_second
                    pbar.set_postfix({"ETA": f"{eta:.2f}s"})
        
        # Run the second half of the trajectory, saving 100 configurations
        with tqdm(total=num_configurations, desc="Second half progress", unit="configs") as pbar:
            start_time = time.time()
            for i in range(num_configurations):
                for j in range(steps_per_configuration):
                    simulation.step(500)
                    gcmc_mover.move(simulation.context, 50)
                    gcmc_mover.report(simulation)
           
                # Save configuration
                state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                positions = state.getPositions()
                box = state.getPeriodicBoxVectors()
           
                # Remove ghosts for this configuration
                ghost_resids = gcmc_mover.getWaterStatusResids(0)
                temp_topology, temp_positions = grand.utils.remove_ghosts(pdb.topology, positions, ghosts=ghost_resids)
           
                # Save configuration to PDB
                with open(f'config_{i+1}.pdb', 'w') as f:
                    PDBFile.writeFile(temp_topology, temp_positions, f, keepIds=True)
                
                pbar.update(1)
                
                elapsed_time = time.time() - start_time
                configs_per_second = (i + 1) / elapsed_time
                eta = (num_configurations - i - 1) / configs_per_second
                pbar.set_postfix({"ETA": f"{eta:.2f}s"})

    # Remove ghosts and write out a PDB
    ghost_resids = gcmc_mover.getWaterStatusResids(0)
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    pdb.topology, pdb.positions = grand.utils.remove_ghosts(pdb.topology, positions,
                                                            ghosts=ghost_resids,
                                                            pdb=f'uvt_{adams_value}.pdb')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with specified Adams value and sampling mode")
    parser.add_argument("-a", "--adams_value", type=float, required=True, help="Adams value for the simulation")
    parser.add_argument("-m", "--mode", type=str, choices=['coarse', 'fine'], required=True, help="Sampling mode: coarse or fine")
    parser.add_argument("-p", "--previous_pdb", type=str, help="Previous configuration in PDB format")
    args = parser.parse_args()

    main(args.adams_value, args.mode, args.previous_pdb)