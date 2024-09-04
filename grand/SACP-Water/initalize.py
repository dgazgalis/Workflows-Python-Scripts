import argparse
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from openmmtools.integrators import BAOABIntegrator
from sys import stdout
import numpy as np
import grand

def main(input_pdb):
    # Create a capped alanine system
    pdb = PDBFile(input_pdb)
    
    # Create a water box
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ForceField('amber14-all.xml', 'amber14/tip3p.xml'), model='tip3p', padding=1.0*nanometers)
    
    # Calculate the dimensions of the system and define new box vectors
    positions = modeller.getPositions()
    pos_array = np.array(positions.value_in_unit(nanometer))
    min_coords = np.min(pos_array, axis=0)
    max_coords = np.max(pos_array, axis=0)
    box_size = max_coords - min_coords

    # Add some padding to the box size
    padding = 0.0  # additional paddin in nanometers
    box_vectors = np.diag(box_size + padding)

    # Set the new box vectors
    modeller.topology.setPeriodicBoxVectors(box_vectors)
    
    # Translate coordinates to ensure they're all positive
    positions_array = np.array(positions.value_in_unit(nanometer))
    min_coords = np.min(positions_array, axis=0)
    positions_array -= min_coords
    modeller.positions = positions_array * nanometer
    
    # Save the solvated system with translated coordinates
    with open('solvated.pdb', 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    # Load the solvated system
    pdb = PDBFile('solvated.pdb')
    
    # Count the number of water molecules
    water_count = sum([1 for residue in pdb.topology.residues() if residue.name == 'HOH'])
    print(f"Number of water molecules: {water_count}")
    
    # Add ghost waters equal to the number of actual water molecules
    pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology, pdb.positions,
                                                             n=water_count, pdb='ghosts.pdb')
    print(f"Added {water_count} ghost waters to match the number of actual water molecules.")
    
    # Count total number of solvent molecules (water + ghosts) and write to solvent_count.log
    total_solvent_count = sum([1 for residue in pdb.topology.residues() if residue.name == 'HOH'])
    with open('solvent_count.log', 'w') as f:
        f.write(f"Total number of solvent molecules (including ghosts): {total_solvent_count}\n")
    
    # Create system
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12*angstrom,
                                     switchDistance=10*angstrom, constraints=HBonds)

    box_vectors = np.array(pdb.topology.getPeriodicBoxVectors())
    
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
    
    # Define GCMC Sampler using StandardGCMCSystemSampler
    gcmc_mover = grand.samplers.StandardGCMCSystemSampler(
        system=system,
        topology=pdb.topology,
        temperature=298*kelvin,
        adams=15,  # Starting value
        boxVectors=box_vectors,
        ghostFile='ghosts.txt',
        log='gcmc.log',
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
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    
    # Prepare the GCMC sampler
    gcmc_mover.initialise(simulation.context, ghosts)
    
    # Run GCMC/MD equilibration (100k GCMC moves over 1 ps - 1000 moves every 10 fs)
    for i in range(500):
        gcmc_mover.move(simulation.context, 5)
        gcmc_mover.report(simulation)
        simulation.step(50)
    
    # Remove ghosts and write out a PDB
    ghost_resids = gcmc_mover.getWaterStatusResids(0)
    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    pdb.topology, pdb.positions = grand.utils.remove_ghosts(pdb.topology, positions,
                                                            ghosts=ghost_resids,
                                                            pdb='uvt_inital.pdb')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with specified input PDB structure")
    parser.add_argument("input_pdb", type=str, help="Input PDB file for the simulation")
    args = parser.parse_args()

    main(args.input_pdb)