#!/usr/bin/env python3
"""
OpenMM NPT Equilibration Script (Refactored)

This script takes a minimized PDB file containing protein and ligand structures,
applies initial velocities at low temperature, gradually heats the system while
releasing protein backbone restraints, performs 1ns heating at target temperature,
and performs NPT equilibration using OpenMM.
It supports custom ligand template files in XML format for proper parameterization.

The script now saves the final system and state XML files by default, which are
required for restarting or continuing simulations. It also creates separate log
files for each simulation phase.

The script follows the project coding standards and provides JSON input/output
capabilities for integration with the custom job scheduler. It saves trajectories
by default, including the initial frame as a PDB and subsequent dynamics as DCD files.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Script version for tracking and reproducibility
SCRIPT_VERSION = "1.7.1" # Added separate log files for each simulation phase.

# Try to import required libraries
try:
    import openmm
    from openmm import app, unit, System
    from openmm.app import PDBFile, Modeller, ForceField, Simulation, DCDReporter, StateDataReporter
    from openmm import LangevinIntegrator, MonteCarloBarostat
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("Warning: OpenMM is not available. Please install it: conda install -c conda-forge openmm")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy is not available. Please install it: conda install -c conda-forge numpy")

# Configuration constants
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Default equilibration parameters (using OpenMM default units)
DEFAULT_INITIAL_TEMP = 50.0 * unit.kelvin
DEFAULT_TARGET_TEMP = 310.0 * unit.kelvin
DEFAULT_PRESSURE = 1.0 * unit.bar
DEFAULT_TIMESTEP = 2.0 * unit.femtoseconds
# EXTENDED HEATING: 5.0 ns / 2 fs = 2,500,000 steps
DEFAULT_HEATING_STEPS = 2500000
# Additional heating at target temperature: 1.0 ns / 2 fs = 500,000 steps
DEFAULT_TARGET_TEMP_HEATING_STEPS = 500000
DEFAULT_EQUILIBRATION_STEPS = 500000
DEFAULT_FRICTION_COEFF = 2.0 / unit.picoseconds  # INCREASED from 1.0 to 2.0 for more aggressive coupling
DEFAULT_BAROSTAT_FREQUENCY = 25

# Restraint parameters - MUCH MORE REASONABLE values
DEFAULT_INITIAL_RESTRAINT_FORCE = 10.0 * unit.kilojoules_per_mole / unit.nanometers**2
DEFAULT_FINAL_RESTRAINT_FORCE = 0.0 * unit.kilojoules_per_mole / unit.nanometers**2
# Release restraints over first half of heating (2.5 ns)
DEFAULT_RESTRAINT_RELEASE_STEPS = 1250000

# Output frequency parameters
DEFAULT_LOG_FREQUENCY = 1000
DEFAULT_TRAJECTORY_INTERVAL = 1000

# Standard amino acid residue names for restraint application
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL'
}

# Backbone atom names for restraints
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


class OpenMMEquilibrator:
    """
    A class to perform NPT equilibration of protein-ligand systems using OpenMM.
    """

    def __init__(self, log_level: str = "INFO", temp_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__) # Logger is now configured in main
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.log_formatter = logging.Formatter(LOG_FORMAT)

        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required but not installed.")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required but not installed.")

        self._temp_files = []

    def setup_logging(self, log_level: str, log_file: Optional[str] = None) -> None:
        """Set up logging to both console and an optional file."""
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.log_formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(self.log_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")

    def _switch_log_file(self, new_log_path: str) -> None:
        """Dynamically switches the target file for the logger."""
        logger = logging.getLogger()
        current_file_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                current_file_handler = handler
                break
        
        if current_file_handler:
            current_file_handler.close()
            logger.removeHandler(current_file_handler)

        if new_log_path:
            new_file_handler = logging.FileHandler(new_log_path, mode='w')
            new_file_handler.setFormatter(self.log_formatter)
            logger.addHandler(new_file_handler)
            self.logger.info(f"Switched logging output to: {new_log_path}")

    def _add_temp_file(self, filepath: str) -> None:
        self._temp_files.append(filepath)

    def cleanup_temp_files(self) -> None:
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Could not clean up temporary file {temp_file}: {e}")
        self._temp_files.clear()

    def validate_inputs(self, pdb_file: str, ligand_templates: List[str] = None) -> bool:
        if not pdb_file or not os.path.exists(pdb_file) or not os.access(pdb_file, os.R_OK):
            self.logger.error(f"PDB file not found, not readable, or path is invalid: {pdb_file}")
            return False

        if ligand_templates:
            for template_file in ligand_templates:
                if not os.path.exists(template_file) or not os.access(template_file, os.R_OK):
                    self.logger.error(f"Ligand template file not found or not readable: {template_file}")
                    return False
        return True

    def load_structure(self, pdb_file: str) -> Tuple[Optional[app.PDBFile], Dict[str, Any]]:
        try:
            self.logger.info(f"Loading minimized PDB structure: {pdb_file}")
            pdb = PDBFile(pdb_file)

            composition = {
                "total_atoms": pdb.topology.getNumAtoms(), "total_residues": pdb.topology.getNumResidues(),
                "protein_residues": 0, "ligand_residues": 0, "water_molecules": 0, "ion_atoms": 0,
                "chains": pdb.topology.getNumChains(), "backbone_atoms": 0, "residue_names": set()
            }

            for residue in pdb.topology.residues():
                resname = residue.name.strip()
                composition["residue_names"].add(resname)

                if resname in STANDARD_RESIDUES:
                    composition["protein_residues"] += 1
                    for atom in residue.atoms():
                        if atom.name in BACKBONE_ATOMS:
                            composition["backbone_atoms"] += 1
                elif resname in ['HOH', 'WAT', 'H2O']:
                    composition["water_molecules"] += 1
                elif resname in ['Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+', 'NA', 'CL']:
                    composition["ion_atoms"] += 1
                else:
                    composition["ligand_residues"] += 1

            self.logger.info(f"Structure loaded: {composition['total_atoms']} atoms, {composition['protein_residues']} protein residues.")
            return pdb, composition
        except Exception as e:
            self.logger.error(f"Error loading PDB structure: {e}")
            return None, {}

    def setup_forcefield(self, protein_ff: str, water_model: str,
                            ligand_templates: List[str] = None) -> Optional[ForceField]:
            try:
                self.logger.info(f"Setting up forcefield: {protein_ff} with {water_model} water")

                protein_forcefields = {
                    "amber14": "amber14-all.xml", 
                    "amber99sb": "amber99sb.xml", 
                    "charmm36": "charmm36.xml",
                    "none": None 
                }
                water_models = {"tip3p": "amber14/tip3pfb.xml", "tip4pew": "amber14/tip4pew.xml", "spce": "amber14/spce.xml"}

                if protein_ff not in protein_forcefields:
                    raise ValueError(f"Unknown protein forcefield: {protein_ff}. Available: {list(protein_forcefields.keys())}")
                if water_model not in water_models:
                    raise ValueError(f"Unknown water model: {water_model}. Available: {list(water_models.keys())}")

                forcefield_files = []
                
                # Add protein forcefield only if not "none"
                if protein_ff != "none":
                    forcefield_files.append(protein_forcefields[protein_ff])
                else:
                    self.logger.info("No protein forcefield specified - using only ligand templates and water model")
                
                # Add water model
                forcefield_files.append(water_models[water_model])
                
                if ligand_templates:
                    forcefield_files.extend(ligand_templates)
                    self.logger.info(f"Added ligand templates: {ligand_templates}")
                elif protein_ff == "none":
                    self.logger.warning("No protein forcefield and no ligand templates specified. "
                                    "This may cause errors if non-water/ion residues are present.")

                self.logger.info(f"Loading forcefield files: {forcefield_files}")
                forcefield = ForceField(*forcefield_files)
                self.logger.info("Forcefield setup completed successfully")
                return forcefield
            except Exception as e:
                self.logger.error(f"Error setting up forcefield: {e}")
                return None

    def create_system_with_restraints(self, pdb: app.PDBFile, forcefield: ForceField,
                                      pressure: unit.Quantity, restraint_force: unit.Quantity,
                                      target_temp: unit.Quantity) -> Optional[Tuple[System, Dict[int, str]]]:
        try:
            self.logger.info(f"Creating system with restraints (force constant: {restraint_force})")
            system = forcefield.createSystem(
                pdb.topology, nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds
            )

            self.logger.info(f"Creating barostat with T={target_temp}")
            # The barostat is initialized with the final target temperature.
            # During the heating phase, this temperature will be dynamically updated in the context.
            barostat = MonteCarloBarostat(pressure, target_temp, DEFAULT_BAROSTAT_FREQUENCY)
            system.addForce(barostat)

            restraint_force_obj = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            restraint_force_obj.addGlobalParameter("k", restraint_force)
            restraint_force_obj.addPerParticleParameter("x0")
            restraint_force_obj.addPerParticleParameter("y0")
            restraint_force_obj.addPerParticleParameter("z0")

            restraint_info = {}
            for residue in pdb.topology.residues():
                if residue.name.strip() in STANDARD_RESIDUES:
                    for atom in residue.atoms():
                        if atom.name in BACKBONE_ATOMS:
                            position = pdb.positions[atom.index]
                            restraint_force_obj.addParticle(atom.index, [position[0], position[1], position[2]])
                            restraint_info[atom.index] = f"{residue.name}-{residue.id}-{atom.name}"

            system.addForce(restraint_force_obj)
            self.logger.info(f"Applied restraints to {len(restraint_info)} backbone atoms")
            return system, restraint_info
        except Exception as e:
            self.logger.error(f"Error creating system with restraints: {e}")
            return None, None

    def create_simulation(self, pdb: app.PDBFile, system: System, temperature: unit.Quantity,
                          timestep: unit.Quantity, friction_coeff: unit.Quantity) -> Optional[Simulation]:
        try:
            self.logger.info(f"Creating simulation (T={temperature}, dt={timestep}, friction={friction_coeff})")
            integrator = LangevinIntegrator(temperature, friction_coeff, timestep)
            simulation = Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            simulation.context.setVelocitiesToTemperature(temperature)
            self.logger.info("Simulation created successfully")
            return simulation
        except Exception as e:
            self.logger.error(f"Error creating simulation: {e}")
            return None

    def update_restraint_force(self, simulation: Simulation, new_force_constant: unit.Quantity) -> bool:
        try:
            # In OpenMM, CustomExternalForce is a type of Force. We iterate to find it.
            # This is a safe way to update parameters without knowing the force index.
            for force in simulation.system.getForces():
                if isinstance(force, openmm.CustomExternalForce):
                    simulation.context.setParameter("k", new_force_constant)
                    self.logger.debug(f"Updated restraint force constant to {new_force_constant}")
                    return True
            self.logger.warning("Could not find restraint force to update")
            return False
        except Exception as e:
            self.logger.error(f"Error updating restraint force: {e}")
            return False

    def collect_simulation_data(self, simulation: Simulation, step: int, total_mass: unit.Quantity,
                                  simulation_data: Dict[str, List]) -> None:
        try:
            state = simulation.context.getState(getEnergy=True, getPositions=True, getVelocities=False)
            potential_energy = state.getPotentialEnergy()
            kinetic_energy = state.getKineticEnergy()

            num_atoms = simulation.topology.getNumAtoms()
            temperature = (2 * kinetic_energy / (3 * num_atoms * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))

            box_vectors = state.getPeriodicBoxVectors()
            volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2] if box_vectors else 0 * unit.nanometers**3
            density = total_mass / volume if volume > 0*unit.nanometer**3 else 0 * unit.gram / unit.centimeters**3

            simulation_data["timesteps"].append(step)
            simulation_data["potential_energies"].append(potential_energy)
            simulation_data["kinetic_energies"].append(kinetic_energy)
            simulation_data["temperatures"].append(temperature)
            simulation_data["volumes"].append(volume)
            simulation_data["densities"].append(density)
            simulation_data["pressures"].append(0 * unit.bar) # Placeholder, barostat pressure is not directly reported this way
        except Exception as e:
            self.logger.debug(f"Could not collect simulation data at step {step}: {e}")

    def run_heating_phase(self, simulation: Simulation, system: System, params: Dict[str, Any],
                          stats: Dict[str, Any], simulation_data: Dict[str, List],
                          traj_file: Optional[str] = None, traj_interval: Optional[int] = None,
                          log_file: Optional[str] = None) -> bool:
        """Runs the heating and restraint release phase, followed by 1ns heating at target temperature."""
        reporter = None
        state_reporter = None
        try:
            initial_temp, target_temp = params['initial_temp'], params['target_temp']
            heating_steps, release_steps = params['heating_steps'], params['restraint_release_steps']
            target_temp_heating_steps = params['target_temp_heating_steps']
            initial_force, final_force = params['initial_restraint_force'], params['final_restraint_force']
            total_mass = stats['total_mass']

            total_heating_time_ns = (heating_steps + target_temp_heating_steps) * params['timestep'].value_in_unit(unit.femtoseconds) / 1e6
            self.logger.info(f"Starting heating phase: {initial_temp} -> {target_temp} over {heating_steps} steps (5.0 ns)")
            self.logger.info(f"Then 1ns heating at {target_temp} over {target_temp_heating_steps} steps")
            self.logger.info(f"Total heating time: {total_heating_time_ns:.1f} ns")
            self.logger.info(f"Restraint release: {initial_force} -> {final_force} over {release_steps} steps (2.5 ns)")
            self.logger.info(f"AGGRESSIVE MODE: 2.0/ps friction, 2 ps per temperature step")

            if traj_file and traj_interval:
                self.logger.info(f"Setting up DCD trajectory for heating: {traj_file} (interval: {traj_interval} steps)")
                reporter = DCDReporter(traj_file, traj_interval)
                simulation.reporters.append(reporter)

            # Add StateDataReporter for detailed thermodynamic monitoring
            if log_file:
                heating_log_file = log_file.replace('.log', '_state.log')
                self.logger.info(f"Setting up StateDataReporter for heating: {heating_log_file}")
                try:
                    state_reporter = StateDataReporter(
                        heating_log_file, 
                        reportInterval=1000,  # Report every 1000 steps (2 ps)
                        step=True,
                        time=True,
                        potentialEnergy=True,
                        kineticEnergy=True,
                        totalEnergy=True,
                        temperature=True,
                        volume=True,
                        density=True,
                        speed=True
                    )
                    simulation.reporters.append(state_reporter)
                    self.logger.info(f"StateDataReporter added successfully for heating")
                except Exception as e:
                    self.logger.warning(f"Failed to create StateDataReporter for heating: {e}")
                    state_reporter = None

            self.collect_simulation_data(simulation, 0, total_mass, simulation_data)
            stats["initial_potential_energy"] = simulation_data["potential_energies"][0]
            stats["initial_kinetic_energy"] = simulation_data["kinetic_energies"][0]
            stats["initial_volume"] = simulation_data["volumes"][0]

            # ========== PHASE 1: GRADUAL HEATING ==========
            self.logger.info(f"PHASE 1: Gradual heating {initial_temp} -> {target_temp}")
            
            # MORE AGGRESSIVE HEATING PROTOCOL: 
            # - Fewer, larger temperature steps with less equilibration time per step
            # - This should help the actual temperature track the set temperature better
            temp_change_interval = 1000  # Steps between temperature changes (2 ps per temperature) - REDUCED from 2500
            num_temp_changes = heating_steps // temp_change_interval
            
            self.logger.info(f"Using aggressive heating: {num_temp_changes} temperature changes, {temp_change_interval} steps ({temp_change_interval * 2:.1f} ps) each")

            for temp_step in range(num_temp_changes):
                # Calculate current temperature for this segment
                progress = temp_step / (num_temp_changes - 1) if num_temp_changes > 1 else 1.0
                current_temp = initial_temp + (target_temp - initial_temp) * progress
                
                # Set the new temperature for integrator and barostat
                simulation.integrator.setTemperature(current_temp)
                simulation.context.setParameter(openmm.MonteCarloBarostat.Temperature(), current_temp)
                
                # Run multiple steps at this temperature to allow equilibration
                for substep in range(temp_change_interval):
                    global_step = temp_step * temp_change_interval + substep
                    
                    # Update restraint force if we're still in the release phase
                    if global_step < release_steps:
                        release_progress = global_step / release_steps
                        current_force = initial_force - (initial_force - final_force) * release_progress
                        self.update_restraint_force(simulation, current_force)

                    simulation.step(1)

                    # Logging and data collection (adjusted for more frequent temperature changes)
                    if (global_step + 1) % (DEFAULT_LOG_FREQUENCY * 20) == 0:  # Log every 20k steps
                        state = simulation.context.getState(getEnergy=True)
                        # Calculate actual temperature from kinetic energy
                        ke = state.getKineticEnergy()
                        
                        # CORRECTED: Use a more accurate degrees of freedom calculation
                        # For HBonds constraints, DOF is approximately 3N - number_of_constraints
                        # But OpenMM doesn't easily give us the exact constraint count
                        # So we'll use a scaling factor based on typical molecular systems with HBonds
                        num_atoms = simulation.topology.getNumAtoms()
                        # Estimate: HBonds typically reduces DOF by ~15-20% for typical proteins
                        estimated_dof = num_atoms * 3 * 0.82  # More realistic estimate
                        approx_temp = (2 * ke / (estimated_dof * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
                        
                        temp_val = current_temp.value_in_unit(unit.kelvin)
                        approx_temp_val = approx_temp.value_in_unit(unit.kelvin)
                        pe_val = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                        ke_val = ke.value_in_unit(unit.kilojoules_per_mole)

                        self.logger.info(f"Gradual heating step {global_step+1}/{heating_steps}: T_set={temp_val:.1f} K, T_approx={approx_temp_val:.1f} K, PE={pe_val:.1f} kJ/mol, KE={ke_val:.1f} kJ/mol")

                        if global_step < release_steps:
                            force_val = current_force.value_in_unit(unit.kilojoules_per_mole / unit.nanometers**2)
                            self.logger.debug(f"   Restraint force: {force_val:.1f} kJ/mol/nm^2")

                    if (global_step + 1) % (DEFAULT_LOG_FREQUENCY * 10) == 0:  # Collect data every 10k steps
                        self.collect_simulation_data(simulation, global_step + 1, total_mass, simulation_data)

            # Handle any remaining steps if heating_steps doesn't divide evenly
            remaining_steps = heating_steps - (num_temp_changes * temp_change_interval)
            if remaining_steps > 0:
                self.logger.info(f"Running final {remaining_steps} gradual heating steps at {target_temp} (final {remaining_steps * 2:.1f} ps)")
                simulation.integrator.setTemperature(target_temp)
                simulation.context.setParameter(openmm.MonteCarloBarostat.Temperature(), target_temp)
                
                for final_step in range(remaining_steps):
                    global_step = num_temp_changes * temp_change_interval + final_step
                    simulation.step(1)

            # DEBUG: Check temperature at end of gradual heating phase
            end_gradual_heating_state = simulation.context.getState(getEnergy=True)
            end_gradual_heating_ke = end_gradual_heating_state.getKineticEnergy()
            end_gradual_heating_temp = (2 * end_gradual_heating_ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
            integrator_temp = simulation.integrator.getTemperature()
            self.logger.info(f"End of gradual heating phase - Integrator: {integrator_temp}, Actual: {end_gradual_heating_temp.value_in_unit(unit.kelvin):.1f} K")
            
            # ========== PHASE 2: 1NS HEATING AT TARGET TEMPERATURE ==========
            self.logger.info(f"PHASE 2: 1ns heating at target temperature {target_temp}")
            
            # Ensure we're at target temperature
            simulation.integrator.setTemperature(target_temp)
            simulation.context.setParameter(openmm.MonteCarloBarostat.Temperature(), target_temp)
            
            # Ensure restraints are fully released (should already be at 0)
            self.update_restraint_force(simulation, final_force)
            
            # Run 1ns at target temperature
            for step in range(target_temp_heating_steps):
                global_step = heating_steps + step
                simulation.step(1)
                
                # Logging and data collection
                if (step + 1) % (DEFAULT_LOG_FREQUENCY * 10) == 0:  # Log every 10k steps
                    state = simulation.context.getState(getEnergy=True)
                    ke = state.getKineticEnergy()
                    
                    # CORRECTED: Use more accurate degrees of freedom calculation
                    num_atoms = simulation.topology.getNumAtoms()
                    estimated_dof = num_atoms * 3 * 0.82  # Account for HBonds constraints
                    approx_temp = (2 * ke / (estimated_dof * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
                    
                    temp_val = target_temp.value_in_unit(unit.kelvin)
                    approx_temp_val = approx_temp.value_in_unit(unit.kelvin)
                    pe_val = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    ke_val = ke.value_in_unit(unit.kilojoules_per_mole)

                    progress_ps = step * 2  # 2 fs timestep
                    self.logger.info(f"Target temp heating {progress_ps:.0f}/{target_temp_heating_steps*2:.0f} ps: T_set={temp_val:.1f} K, T_approx={approx_temp_val:.1f} K, PE={pe_val:.1f} kJ/mol, KE={ke_val:.1f} kJ/mol")
                    
                    # Also log OpenMM's direct temperature calculation for comparison
                    omm_temp = simulation.integrator.getTemperature().value_in_unit(unit.kelvin)
                    self.logger.info(f"   OpenMM integrator temp: {omm_temp:.1f} K, Approx. calculation: {approx_temp_val:.1f} K")

                if (step + 1) % (DEFAULT_LOG_FREQUENCY * 5) == 0:  # Collect data every 5k steps
                    self.collect_simulation_data(simulation, global_step + 1, total_mass, simulation_data)

            # DEBUG: Check temperature at end of target temperature heating phase
            end_target_heating_state = simulation.context.getState(getEnergy=True)
            end_target_heating_ke = end_target_heating_state.getKineticEnergy()
            end_target_heating_temp = (2 * end_target_heating_ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
            self.logger.info(f"End of target temperature heating phase - Actual: {end_target_heating_temp.value_in_unit(unit.kelvin):.1f} K")
            
            self.logger.info("Heating phase (gradual + target temperature) completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error during heating phase: {e}", exc_info=True)
            return False
        finally:
            if reporter and reporter in simulation.reporters:
                simulation.reporters.remove(reporter)
            if state_reporter and state_reporter in simulation.reporters:
                simulation.reporters.remove(state_reporter)

    def run_equilibration_phase(self, simulation: Simulation, params: Dict[str, Any],
                                stats: Dict[str, Any], simulation_data: Dict[str, List],
                                traj_file: Optional[str] = None, traj_interval: Optional[int] = None,
                                log_file: Optional[str] = None) -> bool:
        """Runs the constant pressure and temperature equilibration phase."""
        reporter = None
        state_reporter = None
        try:
            target_temp, equilibration_steps = params['target_temp'], params['equilibration_steps']
            heating_steps = params['heating_steps']
            target_temp_heating_steps = params['target_temp_heating_steps']
            total_mass = stats['total_mass']

            self.logger.info(f"Starting NPT equilibration at {target_temp} for {equilibration_steps} steps (1 ns)")
            
            # --- CORRECTION: Ensure final equilibration is fully unrestrained ---
            self.logger.info("Setting final restraint force to 0.0 for unrestrained equilibration.")
            self.update_restraint_force(simulation, 0.0 * unit.kilojoules_per_mole / unit.nanometers**2)
            
            # Verify the heating actually worked by checking current temperature
            pre_equil_state = simulation.context.getState(getEnergy=True)
            pre_equil_ke = pre_equil_state.getKineticEnergy()
            pre_equil_temp = (2 * pre_equil_ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
            self.logger.info(f"Starting equilibration - Current temperature: {pre_equil_temp.value_in_unit(unit.kelvin):.1f} K")
            
            # Set integrator and barostat temperatures (should already be correct from heating)
            simulation.integrator.setTemperature(target_temp)
            simulation.context.setParameter(openmm.MonteCarloBarostat.Temperature(), target_temp)
            
            # Only reassign velocities if the current temperature is significantly off
            temp_diff = abs(pre_equil_temp.value_in_unit(unit.kelvin) - target_temp.value_in_unit(unit.kelvin))
            if temp_diff > 20:
                self.logger.info(f"Temperature difference {temp_diff:.1f} K > 20 K, reassigning velocities")
                simulation.context.setVelocitiesToTemperature(target_temp)
                # Verify velocity reassignment worked
                post_velocity_state = simulation.context.getState(getEnergy=True)
                post_velocity_ke = post_velocity_state.getKineticEnergy()
                post_velocity_temp = (2 * post_velocity_ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
                self.logger.info(f"After velocity reassignment - Temperature: {post_velocity_temp.value_in_unit(unit.kelvin):.1f} K")
            else:
                self.logger.info(f"Temperature difference {temp_diff:.1f} K acceptable, keeping current velocities")

            if traj_file and traj_interval:
                self.logger.info(f"Setting up DCD trajectory for equilibration: {traj_file} (interval: {traj_interval} steps)")
                reporter = DCDReporter(traj_file, traj_interval)
                simulation.reporters.append(reporter)

            # Add StateDataReporter for detailed thermodynamic monitoring
            if log_file:
                equilibration_log_file = log_file.replace('.log', '_state.log')
                self.logger.info(f"Setting up StateDataReporter for equilibration: {equilibration_log_file}")
                try:
                    state_reporter = StateDataReporter(
                        equilibration_log_file,
                        reportInterval=1000,  # Report every 1000 steps (2 ps)
                        step=True,
                        time=True,
                        potentialEnergy=True,
                        kineticEnergy=True,
                        totalEnergy=True,
                        temperature=True,
                        volume=True,
                        density=True,
                        progress=True,
                        remainingTime=True,
                        speed=True,
                        elapsedTime=True,
                        separator='\t'
                    )
                    simulation.reporters.append(state_reporter)
                    self.logger.info(f"StateDataReporter added successfully for equilibration")
                except Exception as e:
                    self.logger.warning(f"Failed to create StateDataReporter for equilibration: {e}")
                    state_reporter = None

            equil_temps, equil_vols = [], []

            for step in range(equilibration_steps):
                simulation.step(1)

                if (step + 1) % (DEFAULT_LOG_FREQUENCY * 1) == 0: # Collect stats every 1000 steps
                    state = simulation.context.getState(getEnergy=True, getPositions=True)
                    ke = state.getKineticEnergy()
                    temp = (2 * ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
                    equil_temps.append(temp)

                    bv = state.getPeriodicBoxVectors()
                    vol = (bv[0][0] * bv[1][1] * bv[2][2]) if bv else 0*unit.nanometer**3
                    equil_vols.append(vol)

                if (step + 1) % (DEFAULT_LOG_FREQUENCY * 10) == 0: # Log every 10000 steps
                    if len(equil_temps) > 10:
                        avg_temp = np.mean([t.value_in_unit(unit.kelvin) for t in equil_temps[-100:]])
                        # Also show the target temperature for comparison
                        target_temp_val = target_temp.value_in_unit(unit.kelvin)
                        self.logger.info(f"Equilibration step {step+1}/{equilibration_steps}: <T>_approx={avg_temp:.1f} K (target: {target_temp_val:.1f} K) | Note: Approx. DoF used.")

                if (step + 1) % (DEFAULT_LOG_FREQUENCY * 5) == 0:
                    # Note: step counting includes both heating phases
                    total_step = heating_steps + target_temp_heating_steps + step + 1
                    self.collect_simulation_data(simulation, total_step, total_mass, simulation_data)

            stats.update({
                "average_temperature": np.mean(equil_temps) if equil_temps else None,
                "temperature_fluctuation": (np.std([t.value_in_unit(unit.kelvin) for t in equil_temps]) * unit.kelvin) if equil_temps else None,
                "final_potential_energy": simulation_data["potential_energies"][-1] if simulation_data["potential_energies"] else None,
                "final_kinetic_energy": simulation_data["kinetic_energies"][-1] if simulation_data["kinetic_energies"] else None,
                "final_volume": equil_vols[-1] if equil_vols else None,
                "volume_fluctuation": (np.std([v.value_in_unit(unit.nanometers**3) for v in equil_vols]) * unit.nanometers**3) if equil_vols else None
            })

            if stats.get("final_volume") and stats.get("total_mass"):
                if stats["final_volume"] > 0 * unit.nanometers**3:
                    stats["final_density"] = stats["total_mass"] / stats["final_volume"]

            self.logger.info("NPT equilibration completed successfully")

            if stats.get('average_temperature'):
                avg_temp_val = stats['average_temperature'].value_in_unit(unit.kelvin)
                temp_fluct_val = stats['temperature_fluctuation'].value_in_unit(unit.kelvin)
                self.logger.info(f"   Average temperature: {avg_temp_val:.2f} ± {temp_fluct_val:.2f} K")

            if stats.get('final_volume'):
                vol_val = stats['final_volume'].value_in_unit(unit.nanometers**3)
                self.logger.info(f"   Final volume: {vol_val:.2f} nm^3")

            return True
        except Exception as e:
            self.logger.error(f"Error during equilibration phase: {e}", exc_info=True)
            return False
        finally:
            if reporter and reporter in simulation.reporters:
                simulation.reporters.remove(reporter)
            if state_reporter and state_reporter in simulation.reporters:
                simulation.reporters.remove(state_reporter)

    def save_pdb_frame(self, topology: app.Topology, positions: unit.Quantity, output_file: str) -> bool:
        """Saves a single frame (topology and positions) to a PDB file."""
        try:
            if not output_file:
                raise ValueError("Output PDB file path is empty.")

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Saving PDB frame to: {output_path}")
            with open(output_path, 'w') as f:
                PDBFile.writeFile(topology, positions, f, keepIds=True)
            self.logger.info("PDB frame saved successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving PDB frame to {output_file}: {e}")
            return False

    def save_equilibrated_structure(self, simulation: Simulation, output_file: str) -> bool:
        """Saves the final equilibrated structure from the simulation context."""
        try:
            self.logger.info(f"Saving final equilibrated structure: {output_file}")
            state = simulation.context.getState(getPositions=True)
            return self.save_pdb_frame(simulation.topology, state.getPositions(), output_file)
        except Exception as e:
            self.logger.error(f"Error saving equilibrated structure: {e}")
            return False

    def save_simulation_state(self, simulation: Simulation, output_file: str) -> bool:
        """Saves the complete simulation state to a serialized XML file for restarting."""
        try:
            if not output_file:
                raise ValueError("Output state file path is empty.")
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            self.logger.info(f"Saving simulation state to XML: {output_file}")
            simulation.saveState(output_file)
            self.logger.info("Simulation state saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error saving simulation state XML: {e}")
            return False

    def save_system_xml(self, system: System, output_file: str) -> bool:
        """Saves the OpenMM System object to a serialized XML file."""
        try:
            if not output_file:
                raise ValueError("Output system XML file path is empty.")
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.logger.info(f"Saving system definition to XML: {output_file}")
            serialized_system = openmm.XmlSerializer.serialize(system)
            with open(output_file, 'w') as f:
                f.write(serialized_system)
            self.logger.info("System XML saved successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving system XML to {output_file}: {e}")
            return False

    def run_npt_equilibration(self, **kwargs: Any) -> Dict[str, Any]:
        pdb_file = kwargs['pdb_file']
        output_file = kwargs['output_file']
        output_state_xml = kwargs.get('output_state_xml')
        output_system_xml = kwargs.get('output_system_xml')
        ligand_templates = kwargs.get('ligand_templates', [])
        main_log_file = kwargs.get('main_log_file')

        output_path = Path(output_file)
        output_dir = output_path.parent
        output_stem = output_path.stem

        heating_log_path = output_dir / f"{output_stem}_heating.log"
        equilibration_log_path = output_dir / f"{output_stem}_equilibration.log"
        
        log_files = {
            "main": main_log_file,
            "heating": str(heating_log_path),
            "equilibration": str(equilibration_log_path),
        }

        stats = {}
        simulation_data = {key: [] for key in [
            "timesteps", "potential_energies", "kinetic_energies",
            "temperatures", "volumes", "densities", "pressures"
        ]}

        unit_params = {
            'initial_temp': kwargs['initial_temp'] * unit.kelvin,
            'target_temp': kwargs['target_temp'] * unit.kelvin,
            'pressure': kwargs['pressure'] * unit.bar,
            'timestep': kwargs['timestep'] * unit.femtoseconds,
            'friction_coeff': kwargs['friction_coeff'] / unit.picoseconds,
            'initial_restraint_force': kwargs['initial_restraint_force'] * unit.kilojoules_per_mole/unit.nanometers**2,
            'final_restraint_force': kwargs['final_restraint_force'] * unit.kilojoules_per_mole/unit.nanometers**2,
            'heating_steps': kwargs['heating_steps'],
            'target_temp_heating_steps': kwargs['target_temp_heating_steps'],
            'equilibration_steps': kwargs['equilibration_steps'],
            'restraint_release_steps': kwargs['restraint_release_steps'],
        }

        # Initialize path variables
        heating_traj_path = None
        equilibration_traj_path = None
        initial_pdb_path = None
        
        try:
            self.logger.info(f"Starting NPT equilibration for {pdb_file}")
            
            # Setup output paths based on the final output PDB path
            save_traj = kwargs.get('save_trajectories', True)

            if save_traj:
                initial_pdb_path = output_dir / f"{output_stem}_initial.pdb"
                heating_traj_path = output_dir / f"{output_stem}_heating.dcd"
                equilibration_traj_path = output_dir / f"{output_stem}_equilibration.dcd"
                self.logger.info(f"Trajectory saving is enabled. Initial PDB: {initial_pdb_path}")
            else:
                self.logger.info("Trajectory saving is disabled.")

            if not self.validate_inputs(pdb_file, ligand_templates):
                raise ValueError("Input validation failed")

            pdb, composition = self.load_structure(pdb_file)
            if pdb is None: raise RuntimeError("Failed to load PDB structure")

            # Save the initial PDB frame (Frame 0) if trajectories are enabled
            if save_traj and initial_pdb_path:
                if not self.save_pdb_frame(pdb.topology, pdb.positions, str(initial_pdb_path)):
                    self.logger.warning(f"Failed to save initial PDB frame to {initial_pdb_path}, continuing.")

            forcefield = self.setup_forcefield(kwargs['protein_ff'], kwargs['water_model'], ligand_templates)
            if forcefield is None: raise RuntimeError("Failed to setup forcefield")

            system, restraint_info = self.create_system_with_restraints(
                pdb, forcefield, unit_params['pressure'],
                unit_params['initial_restraint_force'], unit_params['target_temp']
            )
            if system is None: raise RuntimeError("Failed to create system with restraints")

            # Save the system XML if a path was provided
            if output_system_xml:
                if not self.save_system_xml(system, output_system_xml):
                    self.logger.warning(f"Failed to save system XML to {output_system_xml}, but continuing.")

            simulation = self.create_simulation(pdb, system, unit_params['initial_temp'], unit_params['timestep'], unit_params['friction_coeff'])
            if simulation is None: raise RuntimeError("Failed to create simulation")

            initial_mass = 0.0 * unit.dalton
            stats['total_mass'] = sum(
                [atom.element.mass for atom in pdb.topology.atoms() if atom.element is not None],
                initial_mass
            )

            # Switch to heating log and run phase
            self._switch_log_file(str(heating_log_path))
            if not self.run_heating_phase(simulation, system, unit_params, stats, simulation_data,
                                          str(heating_traj_path) if heating_traj_path else None,
                                          kwargs.get('trajectory_interval'),
                                          str(heating_log_path)):
                raise RuntimeError("Heating phase failed")
            
            # Switch to equilibration log and run phase
            self._switch_log_file(str(equilibration_log_path))
            if not self.run_equilibration_phase(simulation, unit_params, stats, simulation_data,
                                                str(equilibration_traj_path) if equilibration_traj_path else None,
                                                kwargs.get('trajectory_interval'),
                                                str(equilibration_log_path)):
                raise RuntimeError("Equilibration phase failed")

            if not self.save_equilibrated_structure(simulation, output_file):
                raise RuntimeError("Failed to save final equilibrated structure")

            if output_state_xml:
                if not self.save_simulation_state(simulation, output_state_xml):
                    self.logger.warning(f"Failed to save simulation state to {output_state_xml}, but continuing as PDB was saved.")

            success = True
            error_message = None

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            success = False
            error_message = str(e)
            composition = {}
            restraint_info = {}

        finally:
            # Switch back to the main log for final messages
            self._switch_log_file(main_log_file)
            self.cleanup_temp_files()

        results = {
            "success": success, "error": error_message, "input_file": pdb_file,
            "initial_pdb_file": str(initial_pdb_path) if success and initial_pdb_path else None,
            "output_file": output_file if success else None,
            "output_state_xml": output_state_xml if success and output_state_xml else None,
            "output_system_xml": output_system_xml if success and output_system_xml else None,
            "heating_trajectory_file": str(heating_traj_path) if success and heating_traj_path else None,
            "equilibration_trajectory_file": str(equilibration_traj_path) if success and equilibration_traj_path else None,
            "log_files": log_files if success else {"main": main_log_file},
            "ligand_templates": ligand_templates, "parameters": kwargs,
            "composition": composition,
            "restraint_info": {"total_restrained_atoms": len(restraint_info), "restraint_details": restraint_info},
            "statistics": stats, "simulation_data": simulation_data
        }

        self.logger.info(f"NPT workflow completed. Success: {success}")
        return results


def _convert_value_for_json(value: Any) -> Any:
    """Recursively converts special objects to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: _convert_value_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_value_for_json(v) for v in value]
    if isinstance(value, set):
        return sorted(list(value))
    if hasattr(value, 'value_in_unit'):
        try:
            # Handle OpenMM unit objects
            return value.value_in_unit(value.unit)
        except AttributeError:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_results_to_json(results: Dict[str, Any], output_file: str,
                         run_options: Dict[str, Any] = None) -> None:
    try:
        results_json = _convert_value_for_json(results)
        stats_summary = results_json.get("statistics", {})

        output_data = {
            "metadata": {"timestamp": datetime.now().isoformat(), "script_version": SCRIPT_VERSION,
                         "openmm_available": OPENMM_AVAILABLE, "numpy_available": NUMPY_AVAILABLE},
            "run_options": _convert_value_for_json(run_options or {}),
            "results": results_json,
            "summary": {
                "equilibration_successful": results_json.get("success", False),
                "total_atoms": results_json.get("composition", {}).get("total_atoms", 0),
                "restrained_atoms": results_json.get("restraint_info", {}).get("total_restrained_atoms", 0),
                "final_temperature_K": stats_summary.get("average_temperature"),
                "temperature_stability": stats_summary.get("temperature_fluctuation"),
                "final_volume_nm3": stats_summary.get("final_volume"),
                "final_density_g_cm3": stats_summary.get("final_density"),
                "initial_pdb_file": results_json.get("initial_pdb_file"),
                "heating_trajectory_file": results_json.get("heating_trajectory_file"),
                "equilibration_trajectory_file": results_json.get("equilibration_trajectory_file"),
            }
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        logging.info(f"Results saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}", exc_info=True)
        raise


def parse_ligand_templates(template_arg: str) -> List[str]:
    if not template_arg: return []
    from glob import glob
    expanded_templates = []
    templates = [t.strip() for t in template_arg.split(',') if t.strip()]
    for template in templates:
        matches = glob(template)
        if matches:
            expanded_templates.extend(matches)
        else:
            logging.warning(f"No files found matching pattern: {template}")
    return expanded_templates

def main():
    parser = argparse.ArgumentParser(
        description=f"OpenMM NPT Equilibration Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python %(prog)s input.pdb output.pdb --ligand-templates "lig.xml"
  python %(prog)s --json-config config.json
  python %(prog)s input.pdb output.pdb -n "my_exp" --trajectory-interval 5000
  python %(prog)s input.pdb output.pdb --no-save-system-xml --no-save-state-xml
  python %(prog)s input.pdb output.pdb --no-run-directory
"""
    )

    parser.add_argument("input_pdb", nargs="?", help="Input minimized PDB file path")
    parser.add_argument("output_pdb", nargs="?", help="Output equilibrated PDB file path")
    parser.add_argument("--ligand-templates", type=str, help="Ligand template XML file(s), comma-separated or glob pattern")
    parser.add_argument("--json-config", type=str, help="JSON configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--protein-ff", default="amber14", choices=["amber14", "amber99sb", "charmm36", "none"])
    parser.add_argument("--water-model", default="tip3p", choices=["tip3p", "tip4pew", "spce"])
    parser.add_argument("--initial-temp", type=float, default=DEFAULT_INITIAL_TEMP.value_in_unit(unit.kelvin))
    parser.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP.value_in_unit(unit.kelvin))
    parser.add_argument("--pressure", type=float, default=DEFAULT_PRESSURE.value_in_unit(unit.bar))
    parser.add_argument("--heating-steps", type=int, default=DEFAULT_HEATING_STEPS)
    parser.add_argument("--target-temp-heating-steps", type=int, default=DEFAULT_TARGET_TEMP_HEATING_STEPS, help=f"Steps for heating at target temperature (default: {DEFAULT_TARGET_TEMP_HEATING_STEPS} = 1ns)")
    parser.add_argument("--equilibration-steps", type=int, default=DEFAULT_EQUILIBRATION_STEPS)
    parser.add_argument("--restraint-release-steps", type=int, default=DEFAULT_RESTRAINT_RELEASE_STEPS)
    parser.add_argument("--initial-restraint-force", type=float, default=DEFAULT_INITIAL_RESTRAINT_FORCE.value_in_unit(unit.kilojoules_per_mole/unit.nanometers**2))
    parser.add_argument("--final-restraint-force", type=float, default=DEFAULT_FINAL_RESTRAINT_FORCE.value_in_unit(unit.kilojoules_per_mole/unit.nanometers**2))
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds))
    parser.add_argument("--friction-coeff", type=float, default=DEFAULT_FRICTION_COEFF.value_in_unit(unit.picosecond**-1))
    parser.add_argument("--save-trajectories", default=True, action=argparse.BooleanOptionalAction, help="Save DCD trajectories and initial PDB. Use --no-save-trajectories to disable.")
    parser.add_argument("--no-save-system-xml", action="store_true", help="Disable saving the final system XML file (saved by default).")
    parser.add_argument("--no-save-state-xml", action="store_true", help="Disable saving the final state XML file (saved by default).")
    parser.add_argument("--trajectory-interval", type=int, default=DEFAULT_TRAJECTORY_INTERVAL, help=f"Interval (in steps) for saving trajectory frames (default: {DEFAULT_TRAJECTORY_INTERVAL}).")
    parser.add_argument("--output-json", type=str, help="Save results to a specific JSON file (name only)")
    parser.add_argument("--workflow-name", type=str, default="equilibration", help="Workflow name for directory naming (default: equilibration)")
    parser.add_argument("-n", "--name", dest="run_name", type=str, help="Run name prefix for directory")
    parser.add_argument("--no-run-directory", action="store_true", help="Don't create timestamped run directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--version", action="version", version=f"%(prog)s v{SCRIPT_VERSION}")

    args = parser.parse_args()

    config = {}
    if args.json_config:
        with open(args.json_config, 'r') as f:
            config = json.load(f)

    # CLI arguments override JSON config
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_args)

    input_pdb = config.get("input_pdb")
    output_pdb = config.get("output_pdb")
    if not input_pdb or not output_pdb:
        parser.error("input_pdb and output_pdb must be provided via command line or JSON config.")

    # --- Directory and Path Setup ---
    run_dir = None
    if not config.get('no_run_directory', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get("run_name")
        workflow_name = config.get("workflow_name", "equilibration")
        dir_name = f"{run_name}_openmm_{workflow_name}_{timestamp}" if run_name else f"openmm_{workflow_name}_{timestamp}"

        base_output_dir = config.get('output_dir', DEFAULT_OUTPUT_DIR)
        run_dir = Path(base_output_dir) / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Make output paths relative to the new run directory
        output_pdb = run_dir / Path(output_pdb).name
        json_output_name = Path(config.get('output_json', 'equilibration_results.json')).name
        config['output_json'] = run_dir / json_output_name
    
    # --- Default XML Path Generation ---
    # This logic runs after the output directory is finalized
    final_output_pdb_path = Path(output_pdb)
    output_stem = final_output_pdb_path.stem

    if not config.get('no_save_system_xml', False):
        config['output_system_xml'] = final_output_pdb_path.with_name(f"{output_stem}_system.xml")
    else:
        config['output_system_xml'] = None

    if not config.get('no_save_state_xml', False):
        config['output_state_xml'] = final_output_pdb_path.with_name(f"{output_stem}_state.xml")
    else:
        config['output_state_xml'] = None

    # --- Logging Setup ---
    main_log_file = run_dir / f"{config.get('workflow_name', 'equilibration')}_main.log" if run_dir else None
    equilibrator = OpenMMEquilibrator(log_level=config.get('log_level', 'INFO'))
    equilibrator.setup_logging(log_level=config.get('log_level', 'INFO'), log_file=str(main_log_file) if main_log_file else None)

    if run_dir:
        equilibrator.logger.info(f"Created run directory: {run_dir}")

    # --- Parameter setup ---
    run_params = {
        'pdb_file': input_pdb,
        'output_file': str(output_pdb),
        'output_state_xml': str(config['output_state_xml']) if config.get('output_state_xml') else None,
        'output_system_xml': str(config['output_system_xml']) if config.get('output_system_xml') else None,
        'main_log_file': str(main_log_file) if main_log_file else None,
        'ligand_templates': parse_ligand_templates(config.get('ligand_templates', '')),
        'protein_ff': config.get('protein_ff', 'amber14'),
        'water_model': config.get('water_model', 'tip3p'),
        'initial_temp': config.get('initial_temp', DEFAULT_INITIAL_TEMP.value_in_unit(unit.kelvin)),
        'target_temp': config.get('target_temp', DEFAULT_TARGET_TEMP.value_in_unit(unit.kelvin)),
        'pressure': config.get('pressure', DEFAULT_PRESSURE.value_in_unit(unit.bar)),
        'heating_steps': config.get('heating_steps', DEFAULT_HEATING_STEPS),
        'target_temp_heating_steps': config.get('target_temp_heating_steps', DEFAULT_TARGET_TEMP_HEATING_STEPS),
        'equilibration_steps': config.get('equilibration_steps', DEFAULT_EQUILIBRATION_STEPS),
        'restraint_release_steps': config.get('restraint_release_steps', DEFAULT_RESTRAINT_RELEASE_STEPS),
        'initial_restraint_force': config.get('initial_restraint_force', DEFAULT_INITIAL_RESTRAINT_FORCE.value_in_unit(unit.kilojoules_per_mole/unit.nanometers**2)),
        'final_restraint_force': config.get('final_restraint_force', DEFAULT_FINAL_RESTRAINT_FORCE.value_in_unit(unit.kilojoules_per_mole/unit.nanometers**2)),
        'timestep': config.get('timestep', DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds)),
        'friction_coeff': config.get('friction_coeff', DEFAULT_FRICTION_COEFF.value_in_unit(unit.picosecond**-1)),
        'save_trajectories': config.get('save_trajectories', True),
        'trajectory_interval': config.get('trajectory_interval', DEFAULT_TRAJECTORY_INTERVAL),
    }

    try:
        results = equilibrator.run_npt_equilibration(**run_params)

        # Add run info to results
        results['run_directory'] = str(run_dir) if run_dir else None
        results['run_name'] = config.get('run_name')
        results['workflow_name'] = config.get('workflow_name', 'equilibration')

        json_output_path = config.get("output_json", "equilibration_results.json")
        save_results_to_json(results, str(json_output_path), run_options=run_params)

        print("\n" + "="*80)
        print("EQUILIBRATION SUMMARY")
        print("="*80)
        if results['success']:
            stats = results['statistics']
            log_files = results.get('log_files', {})
            avg_temp_q = stats.get('average_temperature')
            temp_fluct_q = stats.get('temperature_fluctuation')
            print(f"Success: Equilibration completed for {run_params['pdb_file']}")
            if run_dir:
                print(f"Run Directory: {run_dir}")
            
            print("\n--- Output Files ---")
            if results.get("initial_pdb_file"):
                print(f"   Initial PDB (Frame 0): {results['initial_pdb_file']}")
            print(f"   Final PDB (Equilibrated): {results['output_file']}")
            if results.get("output_system_xml"):
                print(f"   Output System XML: {results['output_system_xml']}")
            if results.get("output_state_xml"):
                print(f"   Output State XML: {results['output_state_xml']}")
            if results.get("heating_trajectory_file"):
                print(f"   Heating Trajectory (DCD): {results['heating_trajectory_file']}")
            if results.get("equilibration_trajectory_file"):
                print(f"   Equilibration Trajectory (DCD): {results['equilibration_trajectory_file']}")
            
            print("\n--- Log Files ---")
            if log_files.get("main"): print(f"   Main Log: {log_files['main']}")
            if log_files.get("heating"): print(f"   Heating Phase Log: {log_files['heating']}")
            if log_files.get("equilibration"): print(f"   Equilibration Phase Log: {log_files['equilibration']}")

            print("\n--- Final Statistics ---")
            if avg_temp_q and temp_fluct_q:
                avg_temp_val = avg_temp_q.value_in_unit(unit.kelvin)
                temp_fluct_val = temp_fluct_q.value_in_unit(unit.kelvin)
                print(f"   Final Temperature: {avg_temp_val:.2f} ± {temp_fluct_val:.2f} K")
            print(f"   Results JSON: {json_output_path}")
            
            # --- DISCLAIMER ---
            print("\n" + "-"*40)
            print("NOTE ON TEMPERATURE REPORTING:")
            print("The 'approximate' temperature values in the main log file are based on a")
            print("simplified degrees-of-freedom calculation. For accurate thermodynamic data")
            print("as calculated by OpenMM, please refer to the '*_state.log' files")
            print("in the run directory.")
            print("-" * 40)
            
        else:
            print(f"Failure: {results.get('error', 'Unknown error')}")
        print("="*80)

        if not results['success']:
            sys.exit(1)

    except Exception as e:
        logging.getLogger(__name__).error(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()