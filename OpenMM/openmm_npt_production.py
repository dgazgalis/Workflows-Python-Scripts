#!/usr/bin/env python3
"""
OpenMM NPT Molecular Dynamics Script with Enhanced Checkpointing

This script performs conventional NPT molecular dynamics simulations using OpenMM.
It takes as input the XML files (system and state) generated from the NPT equilibration
script and runs production MD simulations at constant temperature and pressure.

The script supports configurable simulation length, output frequencies, analysis
options, and comprehensive checkpoint/restart functionality.

Features:
- Restart from equilibration XML files
- Configurable simulation length and output frequencies
- Real-time monitoring and logging
- Trajectory analysis and statistics collection
- JSON input/output for scheduler integration
- Separate log files for different phases
- Enhanced checkpoint and restart system
- Automatic restart script generation
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
import stat
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Script version for tracking and reproducibility
SCRIPT_VERSION = "1.1.0"

# Try to import required libraries
try:
    import openmm
    from openmm import app, unit, System
    from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
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

# Default MD parameters (using OpenMM default units)
DEFAULT_TARGET_TEMP = 310.0 * unit.kelvin
DEFAULT_PRESSURE = 1.0 * unit.bar
DEFAULT_TIMESTEP = 2.0 * unit.femtoseconds
DEFAULT_FRICTION_COEFF = 1.0 / unit.picoseconds
DEFAULT_BAROSTAT_FREQUENCY = 25

# Default simulation length (10 ns)
DEFAULT_MD_STEPS = 5000000  # 10 ns / 2 fs = 5,000,000 steps

# Output frequency parameters
DEFAULT_LOG_FREQUENCY = 1000
DEFAULT_TRAJECTORY_INTERVAL = 5000  # Save frame every 10 ps
DEFAULT_CHECKPOINT_INTERVAL = 500000  # Save checkpoint every 1 ns

# Analysis parameters
DEFAULT_ANALYSIS_STRIDE = 1000  # Analyze every 1000 frames for statistics


class OpenMMProduction:
    """
    A class to perform NPT molecular dynamics simulations using OpenMM with enhanced checkpointing.
    """

    def __init__(self, log_level: str = "INFO", temp_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
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

    def validate_inputs(self, system_xml: str, state_xml: str, 
                       equilibrated_pdb: Optional[str] = None) -> bool:
        """Validate that required input files exist and are readable."""
        if not system_xml or not os.path.exists(system_xml) or not os.access(system_xml, os.R_OK):
            self.logger.error(f"System XML file not found or not readable: {system_xml}")
            return False

        if not state_xml or not os.path.exists(state_xml) or not os.access(state_xml, os.R_OK):
            self.logger.error(f"State XML file not found or not readable: {state_xml}")
            return False

        if equilibrated_pdb and (not os.path.exists(equilibrated_pdb) or not os.access(equilibrated_pdb, os.R_OK)):
            self.logger.error(f"Equilibrated PDB file not found or not readable: {equilibrated_pdb}")
            return False

        return True

    def load_system_and_state(self, system_xml: str, state_xml: str, 
                             equilibrated_pdb: Optional[str] = None) -> Tuple[Optional[System], Optional[openmm.State], Optional[app.Topology]]:
        """Load the OpenMM system and state from XML files."""
        try:
            self.logger.info(f"Loading system from XML: {system_xml}")
            with open(system_xml, 'r') as f:
                system = openmm.XmlSerializer.deserialize(f.read())

            self.logger.info(f"Loading state from XML: {state_xml}")
            # Load the state XML
            with open(state_xml, 'r') as f:
                state = openmm.XmlSerializer.deserialize(f.read())

            # Load topology from PDB if provided
            topology = None
            if equilibrated_pdb:
                self.logger.info(f"Loading topology from PDB: {equilibrated_pdb}")
                pdb = PDBFile(equilibrated_pdb)
                topology = pdb.topology
            else:
                self.logger.warning("No PDB file provided. Topology will need to be set separately.")

            self.logger.info("System, state, and topology loaded successfully")
            return system, state, topology

        except Exception as e:
            self.logger.error(f"Error loading system/state from XML files: {e}")
            return None, None, None

    def create_production_simulation(self, system: System, topology: app.Topology, 
                                   initial_state: openmm.State, temperature: unit.Quantity,
                                   timestep: unit.Quantity, friction_coeff: unit.Quantity) -> Optional[Simulation]:
        """Create a new simulation for production MD run."""
        try:
            self.logger.info(f"Creating production simulation (T={temperature}, dt={timestep}, friction={friction_coeff})")
            
            integrator = LangevinIntegrator(temperature, friction_coeff, timestep)
            simulation = Simulation(topology, system, integrator)
            
            # Set the initial state (positions, velocities, box vectors)
            simulation.context.setState(initial_state)
            
            self.logger.info("Production simulation created successfully")
            return simulation
            
        except Exception as e:
            self.logger.error(f"Error creating production simulation: {e}")
            return None

    def setup_checkpoint_and_state_saving(self, simulation, params, output_dir, output_stem, 
                                          checkpoint_interval=None, state_interval=None):
        """
        Set up comprehensive checkpoint and state saving for restart capability.
        
        Args:
            simulation: OpenMM simulation object
            params: Simulation parameters dictionary
            output_dir: Output directory path
            output_stem: Base name for output files
            checkpoint_interval: Steps between checkpoint saves (default: every 1 ns)
            state_interval: Steps between state XML saves (default: every 2 ns)
        """
        try:
            # Default intervals
            if checkpoint_interval is None:
                checkpoint_interval = params.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL)
            if state_interval is None:
                state_interval = params.get('state_interval', checkpoint_interval * 2)

            # Create checkpoint directory
            checkpoint_dir = Path(output_dir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            # Set up checkpoint reporter (binary format, fast)
            checkpoint_file = checkpoint_dir / f"{output_stem}_latest.chk"
            checkpoint_reporter = app.CheckpointReporter(str(checkpoint_file), checkpoint_interval)
            simulation.reporters.append(checkpoint_reporter)
            
            self.logger.info(f"Checkpoint saving every {checkpoint_interval} steps to: {checkpoint_file}")

            # Store checkpoint info for manual state saving
            self.checkpoint_info = {
                'checkpoint_dir': checkpoint_dir,
                'output_stem': output_stem,
                'state_interval': state_interval,
                'checkpoint_interval': checkpoint_interval,
                'last_state_save': 0,
                'system_xml': params.get('system_xml', '')  # Store system XML path for restart
            }

            return checkpoint_reporter

        except Exception as e:
            self.logger.error(f"Error setting up checkpoint system: {e}")
            return None

    def save_restart_state(self, simulation, step, force_save=False):
        """
        Save complete restart state including XML state and PDB structure.
        
        Args:
            simulation: OpenMM simulation object
            step: Current simulation step
            force_save: Force save regardless of interval
        """
        try:
            if not hasattr(self, 'checkpoint_info'):
                return

            checkpoint_info = self.checkpoint_info
            state_interval = checkpoint_info['state_interval']
            
            # Check if it's time to save state
            if not force_save and (step - checkpoint_info['last_state_save']) < state_interval:
                return

            checkpoint_dir = checkpoint_info['checkpoint_dir']
            output_stem = checkpoint_info['output_stem']

            # Create timestamped state files
            state_xml_file = checkpoint_dir / f"{output_stem}_step_{step}_state.xml"
            structure_pdb_file = checkpoint_dir / f"{output_stem}_step_{step}_structure.pdb"

            # Save state XML (positions, velocities, box vectors, etc.)
            self.logger.info(f"Saving restart state at step {step}")
            simulation.saveState(str(state_xml_file))

            # Save current structure as PDB
            state = simulation.context.getState(getPositions=True)
            with open(structure_pdb_file, 'w') as f:
                PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)

            # Also save as "latest" for easy restart
            latest_state_xml = checkpoint_dir / f"{output_stem}_latest_state.xml"
            latest_structure_pdb = checkpoint_dir / f"{output_stem}_latest_structure.pdb"
            
            # Copy to latest files
            shutil.copy2(state_xml_file, latest_state_xml)
            shutil.copy2(structure_pdb_file, latest_structure_pdb)

            # Create restart info file
            restart_info = {
                'step': step,
                'system_xml': checkpoint_info.get('system_xml', ''),  # Include system XML path
                'state_xml': str(state_xml_file),
                'structure_pdb': str(structure_pdb_file),
                'latest_state_xml': str(latest_state_xml),
                'latest_structure_pdb': str(latest_structure_pdb),
                'checkpoint_file': str(checkpoint_dir / f"{output_stem}_latest.chk"),
                'timestamp': datetime.now().isoformat(),
                'simulation_time_ns': step * DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds) / 1e6
            }

            restart_info_file = checkpoint_dir / f"{output_stem}_restart_info.json"
            with open(restart_info_file, 'w') as f:
                json.dump(restart_info, f, indent=2)

            checkpoint_info['last_state_save'] = step
            self.logger.info(f"Restart state saved: step {step}, time {restart_info['simulation_time_ns']:.2f} ns")

        except Exception as e:
            self.logger.error(f"Error saving restart state at step {step}: {e}")

    def load_restart_state(self, restart_info_file):
        """
        Load restart information from a previous simulation.
        
        Args:
            restart_info_file: Path to restart info JSON file
            
        Returns:
            Dictionary with restart information
        """
        try:
            with open(restart_info_file, 'r') as f:
                restart_info = json.load(f)
            
            # Validate that files exist
            required_files = ['state_xml', 'structure_pdb', 'checkpoint_file']
            for file_key in required_files:
                file_path = restart_info.get(file_key)
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"Restart file not found: {file_path}")
            
            self.logger.info(f"Loaded restart info from step {restart_info['step']}")
            self.logger.info(f"Simulation time: {restart_info['simulation_time_ns']:.2f} ns")
            
            return restart_info
            
        except Exception as e:
            self.logger.error(f"Error loading restart state: {e}")
            return None

    def create_restart_script(self, restart_info_file, output_dir):
        """
        Create a convenient restart script for continuing the simulation.
        """
        try:
            restart_script_content = f'''#!/usr/bin/env python3
"""
Auto-generated restart script for OpenMM simulation.
Generated on: {datetime.now().isoformat()}
"""

import sys
import os
from pathlib import Path

# Add the script directory to path to import the main module
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import the main OpenMM production module
from openmm_production import OpenMMProduction, save_results_to_json

def main():
    # Load restart information
    restart_info_file = "{restart_info_file}"
    
    print(f"Restarting simulation from: {{restart_info_file}}")
    
    # Initialize production runner
    production_runner = OpenMMProduction(log_level="INFO")
    
    # Load restart info
    restart_info = production_runner.load_restart_state(restart_info_file)
    if not restart_info:
        print("Failed to load restart information!")
        sys.exit(1)
    
    print(f"Continuing from step {{restart_info['step']}}")
    print(f"Previous simulation time: {{restart_info['simulation_time_ns']:.2f}} ns")
    
    # Set up parameters for continuation
    # You may want to modify these parameters for the continuation run
    run_params = {{
        'system_xml': restart_info['state_xml'],  # Use saved state as system
        'state_xml': restart_info['state_xml'],
        'equilibrated_pdb': restart_info['structure_pdb'],
        'output_pdb': 'continued_simulation_final.pdb',
        'main_log_file': 'continued_simulation.log',
        'target_temp': 310.0,  # Adjust as needed
        'pressure': 1.0,
        'timestep': 2.0,
        'friction_coeff': 1.0,
        'md_steps': 2500000,  # Additional 5 ns, adjust as needed
        'save_trajectories': True,
        'save_checkpoints': True,
        'trajectory_interval': 5000,
        'checkpoint_interval': 500000,
    }}
    
    # Run the continuation
    results = production_runner.run_npt_production(**run_params)
    
    # Save results
    save_results_to_json(results, "continued_simulation_results.json", run_options=run_params)
    
    if results['success']:
        print("Simulation continuation completed successfully!")
    else:
        print(f"Simulation continuation failed: {{results.get('error', 'Unknown error')}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
            
            restart_script_path = Path(output_dir) / "restart_simulation.py"
            with open(restart_script_path, 'w') as f:
                f.write(restart_script_content)
            
            # Make it executable
            os.chmod(restart_script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            
            self.logger.info(f"Created restart script: {restart_script_path}")
            return str(restart_script_path)
            
        except Exception as e:
            self.logger.error(f"Error creating restart script: {e}")
            return None

    def analyze_trajectory_statistics(self, simulation_data: Dict[str, List]) -> Dict[str, Any]:
        """Analyze trajectory data to compute statistics."""
        try:
            stats = {}
            
            if simulation_data.get("temperatures"):
                temps = [t.value_in_unit(unit.kelvin) for t in simulation_data["temperatures"]]
                stats["temperature"] = {
                    "mean": np.mean(temps),
                    "std": np.std(temps),
                    "min": np.min(temps),
                    "max": np.max(temps)
                }
            
            if simulation_data.get("potential_energies"):
                pe = [e.value_in_unit(unit.kilojoules_per_mole) for e in simulation_data["potential_energies"]]
                stats["potential_energy"] = {
                    "mean": np.mean(pe),
                    "std": np.std(pe),
                    "min": np.min(pe),
                    "max": np.max(pe)
                }
            
            if simulation_data.get("kinetic_energies"):
                ke = [e.value_in_unit(unit.kilojoules_per_mole) for e in simulation_data["kinetic_energies"]]
                stats["kinetic_energy"] = {
                    "mean": np.mean(ke),
                    "std": np.std(ke),
                    "min": np.min(ke),
                    "max": np.max(ke)
                }
            
            if simulation_data.get("volumes"):
                vols = [v.value_in_unit(unit.nanometers**3) for v in simulation_data["volumes"]]
                stats["volume"] = {
                    "mean": np.mean(vols),
                    "std": np.std(vols),
                    "min": np.min(vols),
                    "max": np.max(vols)
                }
            
            if simulation_data.get("densities"):
                densities = [d.value_in_unit(unit.gram/unit.centimeter**3) for d in simulation_data["densities"]]
                stats["density"] = {
                    "mean": np.mean(densities),
                    "std": np.std(densities),
                    "min": np.min(densities),
                    "max": np.max(densities)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing trajectory statistics: {e}")
            return {}

    def collect_simulation_data(self, simulation: Simulation, step: int, total_mass: unit.Quantity,
                              simulation_data: Dict[str, List]) -> None:
        """Collect simulation data for analysis."""
        try:
            state = simulation.context.getState(getEnergy=True, getPositions=False, getVelocities=False)
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
            simulation_data["total_energies"].append(potential_energy + kinetic_energy)
            
        except Exception as e:
            self.logger.debug(f"Could not collect simulation data at step {step}: {e}")

    def run_production_md_with_checkpoints(self, simulation: Simulation, params: Dict[str, Any],
                                         simulation_data: Dict[str, List], total_mass: unit.Quantity,
                                         traj_file: Optional[str] = None, traj_interval: Optional[int] = None,
                                         log_file: Optional[str] = None) -> bool:
        """Enhanced production MD run with comprehensive checkpoint and restart capability."""
        reporters_to_cleanup = []
        checkpoint_reporter = None
        
        try:
            md_steps = params['md_steps']
            target_temp = params['target_temp']
            simulation_time_ns = md_steps * params['timestep'].value_in_unit(unit.femtoseconds) / 1e6
            
            # Extract output directory and stem from trajectory file
            if traj_file:
                output_path = Path(traj_file)
                output_dir = output_path.parent
                output_stem = output_path.stem.replace('_production', '')
            else:
                output_dir = Path("output")
                output_stem = "simulation"

            self.logger.info(f"Starting production MD with checkpointing")
            self.logger.info(f"Simulation length: {md_steps} steps ({simulation_time_ns:.1f} ns)")
            self.logger.info(f"Temperature: {target_temp}")
            self.logger.info(f"Pressure: {params['pressure']}")
            
            # Set up checkpoint and state saving
            checkpoint_reporter = self.setup_checkpoint_and_state_saving(
                simulation, params, output_dir, output_stem
            )

            # Set up trajectory reporter
            if traj_file and traj_interval:
                self.logger.info(f"Setting up DCD trajectory: {traj_file} (interval: {traj_interval} steps)")
                traj_reporter = DCDReporter(traj_file, traj_interval)
                simulation.reporters.append(traj_reporter)
                reporters_to_cleanup.append(traj_reporter)

            # Set up state data reporter for detailed thermodynamic monitoring
            if log_file:
                state_log_file = log_file.replace('.log', '_state.log')
                self.logger.info(f"Setting up StateDataReporter: {state_log_file}")
                try:
                    state_reporter = StateDataReporter(
                        state_log_file,
                        reportInterval=DEFAULT_LOG_FREQUENCY,
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
                        separator='\t',
                        totalSteps=md_steps  # Fixed: Add totalSteps parameter
                    )
                    simulation.reporters.append(state_reporter)
                    reporters_to_cleanup.append(state_reporter)
                    self.logger.info(f"StateDataReporter added successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to create StateDataReporter: {e}")

            # Collect initial data
            self.collect_simulation_data(simulation, 0, total_mass, simulation_data)
            
            # Get initial state for monitoring
            initial_state = simulation.context.getState(getEnergy=True)
            initial_pe = initial_state.getPotentialEnergy()
            initial_ke = initial_state.getKineticEnergy()
            initial_temp = (2 * initial_ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
            
            self.logger.info(f"Initial conditions:")
            self.logger.info(f"  PE: {initial_pe.value_in_unit(unit.kilojoules_per_mole):.1f} kJ/mol")
            self.logger.info(f"  KE: {initial_ke.value_in_unit(unit.kilojoules_per_mole):.1f} kJ/mol")
            self.logger.info(f"  Temperature: {initial_temp.value_in_unit(unit.kelvin):.1f} K")

            # Save initial restart state
            self.save_restart_state(simulation, 0, force_save=True)

            # Main MD loop with periodic state saving
            log_frequency = params.get('log_frequency', DEFAULT_LOG_FREQUENCY)
            analysis_stride = params.get('analysis_stride', DEFAULT_ANALYSIS_STRIDE)
            state_interval = self.checkpoint_info['state_interval']
            
            for step in range(1, md_steps + 1):
                simulation.step(1)
                
                # Periodic logging
                if step % (log_frequency * 50) == 0:  # Log every 50k steps
                    state = simulation.context.getState(getEnergy=True)
                    pe = state.getPotentialEnergy()
                    ke = state.getKineticEnergy()
                    temp = (2 * ke / (3 * simulation.topology.getNumAtoms() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA))
                    
                    progress_ns = step * params['timestep'].value_in_unit(unit.femtoseconds) / 1e6
                    progress_percent = (step / md_steps) * 100
                    
                    self.logger.info(f"Step {step}/{md_steps} ({progress_percent:.1f}%, {progress_ns:.2f} ns): "
                                   f"T={temp.value_in_unit(unit.kelvin):.1f} K, "
                                   f"PE={pe.value_in_unit(unit.kilojoules_per_mole):.1f} kJ/mol, "
                                   f"KE={ke.value_in_unit(unit.kilojoules_per_mole):.1f} kJ/mol")
                
                # Collect data for analysis
                if step % analysis_stride == 0:
                    self.collect_simulation_data(simulation, step, total_mass, simulation_data)
                
                # Save restart state periodically
                if step % state_interval == 0:
                    self.save_restart_state(simulation, step)

            # Save final restart state
            self.save_restart_state(simulation, md_steps, force_save=True)
            
            self.logger.info("Production MD simulation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during production MD simulation: {e}", exc_info=True)
            return False
            
        finally:
            # Clean up reporters
            for reporter in reporters_to_cleanup:
                if reporter in simulation.reporters:
                    simulation.reporters.remove(reporter)
            if checkpoint_reporter and checkpoint_reporter in simulation.reporters:
                simulation.reporters.remove(checkpoint_reporter)

    def save_final_structure(self, simulation: Simulation, output_file: str) -> bool:
        """Save the final structure from the simulation."""
        try:
            self.logger.info(f"Saving final structure: {output_file}")
            state = simulation.context.getState(getPositions=True)
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)
            
            self.logger.info("Final structure saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving final structure: {e}")
            return False

    def save_final_state(self, simulation: Simulation, output_file: str) -> bool:
        """Save the final simulation state to XML."""
        try:
            if not output_file:
                return True  # Skip if no output file specified
                
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            self.logger.info(f"Saving final simulation state: {output_file}")
            simulation.saveState(output_file)
            self.logger.info("Final simulation state saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving final simulation state: {e}")
            return False

    def calculate_total_mass(self, topology: app.Topology) -> unit.Quantity:
        """Calculate the total mass of the system."""
        try:
            total_mass = 0.0 * unit.dalton
            for atom in topology.atoms():
                if atom.element is not None:
                    total_mass += atom.element.mass
            return total_mass
        except Exception as e:
            self.logger.error(f"Error calculating total mass: {e}")
            return 0.0 * unit.dalton

    def run_npt_production(self, **kwargs: Any) -> Dict[str, Any]:
        """Main method to run NPT production MD simulation with checkpointing."""
        system_xml = kwargs['system_xml']
        state_xml = kwargs['state_xml']
        equilibrated_pdb = kwargs.get('equilibrated_pdb')
        output_pdb = kwargs['output_pdb']
        output_state_xml = kwargs.get('output_state_xml')
        main_log_file = kwargs.get('main_log_file')

        output_path = Path(output_pdb)
        output_dir = output_path.parent
        output_stem = output_path.stem

        production_log_path = output_dir / f"{output_stem}_production.log"
        
        log_files = {
            "main": main_log_file,
            "production": str(production_log_path),
        }

        stats = {}
        simulation_data = {key: [] for key in [
            "timesteps", "potential_energies", "kinetic_energies",
            "temperatures", "volumes", "densities", "total_energies"
        ]}

        unit_params = {
            'target_temp': kwargs['target_temp'] * unit.kelvin,
            'pressure': kwargs['pressure'] * unit.bar,
            'timestep': kwargs['timestep'] * unit.femtoseconds,
            'friction_coeff': kwargs['friction_coeff'] / unit.picoseconds,
            'md_steps': kwargs['md_steps'],
            'log_frequency': kwargs.get('log_frequency', DEFAULT_LOG_FREQUENCY),
            'analysis_stride': kwargs.get('analysis_stride', DEFAULT_ANALYSIS_STRIDE),
            'checkpoint_interval': kwargs.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL),
            'state_interval': kwargs.get('state_interval', kwargs.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL) * 2),
        }

        # Initialize path variables
        traj_path = None
        checkpoint_path = None
        
        try:
            self.logger.info(f"Starting NPT production MD")
            self.logger.info(f"Input system XML: {system_xml}")
            self.logger.info(f"Input state XML: {state_xml}")
            if equilibrated_pdb:
                self.logger.info(f"Input PDB: {equilibrated_pdb}")

            # Setup output paths
            save_traj = kwargs.get('save_trajectories', True)
            save_checkpoints = kwargs.get('save_checkpoints', True)

            if save_traj:
                traj_path = output_dir / f"{output_stem}_production.dcd"
                self.logger.info(f"Trajectory saving enabled: {traj_path}")

            if save_checkpoints:
                checkpoint_path = output_dir / "checkpoints" / f"{output_stem}_latest.chk"
                self.logger.info(f"Checkpoint saving enabled: {checkpoint_path}")

            # Validate inputs
            if not self.validate_inputs(system_xml, state_xml, equilibrated_pdb):
                raise ValueError("Input validation failed")

            # Load system, state, and topology
            system, initial_state, topology = self.load_system_and_state(
                system_xml, state_xml, equilibrated_pdb
            )
            if system is None or initial_state is None or topology is None:
                raise RuntimeError("Failed to load system, state, or topology")

            # Calculate total mass
            total_mass = self.calculate_total_mass(topology)
            stats['total_mass'] = total_mass

            # Create production simulation
            simulation = self.create_production_simulation(
                system, topology, initial_state, unit_params['target_temp'],
                unit_params['timestep'], unit_params['friction_coeff']
            )
            if simulation is None:
                raise RuntimeError("Failed to create production simulation")

            # Switch to production log and run simulation with enhanced checkpointing
            self._switch_log_file(str(production_log_path))
            
            if not self.run_production_md_with_checkpoints(
                simulation, unit_params, simulation_data, total_mass,
                str(traj_path) if traj_path else None,
                kwargs.get('trajectory_interval', DEFAULT_TRAJECTORY_INTERVAL),
                str(production_log_path)
            ):
                raise RuntimeError("Production MD simulation failed")

            # Analyze trajectory statistics
            trajectory_stats = self.analyze_trajectory_statistics(simulation_data)
            stats.update(trajectory_stats)

            # Save final structure
            if not self.save_final_structure(simulation, output_pdb):
                raise RuntimeError("Failed to save final structure")

            # Save final state if requested
            if output_state_xml:
                if not self.save_final_state(simulation, output_state_xml):
                    self.logger.warning(f"Failed to save final state to {output_state_xml}")

            # Create restart script for easy continuation
            if hasattr(self, 'checkpoint_info'):
                restart_info_file = self.checkpoint_info['checkpoint_dir'] / f"{output_stem}_restart_info.json"
                self.create_restart_script(restart_info_file, output_dir)

            success = True
            error_message = None

        except Exception as e:
            self.logger.error(f"Production MD workflow failed: {e}")
            success = False
            error_message = str(e)

        finally:
            # Switch back to main log
            self._switch_log_file(main_log_file)
            self.cleanup_temp_files()

        # Prepare results
        results = {
            "success": success,
            "error": error_message,
            "input_system_xml": system_xml,
            "input_state_xml": state_xml,
            "input_pdb": equilibrated_pdb,
            "output_pdb": output_pdb if success else None,
            "output_state_xml": output_state_xml if success and output_state_xml else None,
            "trajectory_file": str(traj_path) if success and traj_path else None,
            "checkpoint_file": str(checkpoint_path) if success and checkpoint_path else None,
            "log_files": log_files if success else {"main": main_log_file},
            "parameters": kwargs,
            "statistics": stats,
            "simulation_data": simulation_data,
            "trajectory_analysis": trajectory_stats if success else {}
        }

        self.logger.info(f"NPT production MD workflow completed. Success: {success}")
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
    """Save results to a JSON file."""
    try:
        results_json = _convert_value_for_json(results)
        stats_summary = results_json.get("statistics", {})

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "script_version": SCRIPT_VERSION,
                "openmm_available": OPENMM_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE
            },
            "run_options": _convert_value_for_json(run_options or {}),
            "results": results_json,
            "summary": {
                "production_successful": results_json.get("success", False),
                "simulation_length_ns": (results_json.get("parameters", {}).get("md_steps", 0) * 
                                       results_json.get("parameters", {}).get("timestep", 2.0) / 1e6),
                "final_temperature": stats_summary.get("temperature", {}).get("mean"),
                "temperature_stability": stats_summary.get("temperature", {}).get("std"),
                "final_volume": stats_summary.get("volume", {}).get("mean"),
                "final_density": stats_summary.get("density", {}).get("mean"),
                "trajectory_file": results_json.get("trajectory_file"),
                "checkpoint_file": results_json.get("checkpoint_file"),
            }
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}", exc_info=True)
        raise


def handle_restart_mode(args):
    """Handle simulation restart from checkpoint."""
    print(f"Restart mode: Loading from {args.restart}")
    
    # Initialize production runner
    production_runner = OpenMMProduction(log_level=args.log_level)
    production_runner.setup_logging(log_level=args.log_level)
    
    # Load restart information
    restart_info = production_runner.load_restart_state(args.restart)
    if not restart_info:
        print("Failed to load restart information!")
        sys.exit(1)
    
    print(f"Continuing from step {restart_info['step']}")
    print(f"Previous simulation time: {restart_info['simulation_time_ns']:.2f} ns")

    # Handle missing system_xml (for old restart files created before this fix)
    if not restart_info.get('system_xml'):
        print("Warning: system_xml not found in restart info. Attempting to locate...")
        restart_dir = Path(args.restart).parent.parent  # Go up from checkpoints/
        # Try to find equilibration directory with matching job name
        job_name = restart_dir.name.split('_openmm_production_')[0]
        equilibration_dirs = list(restart_dir.parent.glob(f"{job_name}_openmm_equilibration_*"))
        if equilibration_dirs:
            system_xml_path = equilibration_dirs[0] / "equilibrated_system.xml"
            if system_xml_path.exists():
                restart_info['system_xml'] = str(system_xml_path)
                print(f"Found system XML: {system_xml_path}")
            else:
                print(f"ERROR: Could not find system XML at {system_xml_path}")
                sys.exit(1)
        else:
            print(f"ERROR: Could not locate equilibration directory for job: {job_name}")
            sys.exit(1)

    # Set up directory for continued run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    restart_dir = Path(args.restart).parent.parent  # Go up from checkpoints/
    continued_dir = restart_dir.parent / f"continued_{timestamp}"
    continued_dir.mkdir(parents=True, exist_ok=True)

    print(f"Continuation output directory: {continued_dir}")

    # Calculate remaining steps to reach original target
    original_target_steps = 250000000  # Original target was 250M steps (500 ns)
    current_step = restart_info['step']
    remaining_steps = original_target_steps - current_step

    # Use remaining steps if continue_steps is the default, otherwise use user-specified value
    if args.continue_steps == DEFAULT_MD_STEPS:
        steps_to_run = remaining_steps
        print(f"Continuing to original target: {original_target_steps} steps")
    else:
        steps_to_run = args.continue_steps
        print(f"Running user-specified additional steps: {args.continue_steps}")

    additional_time_ns = steps_to_run * args.timestep / 1e6
    print(f"Running {steps_to_run} steps ({additional_time_ns:.1f} ns)")
    
    run_params = {
        'system_xml': restart_info.get('system_xml', restart_info['state_xml']),  # Use original system XML
        'state_xml': restart_info['state_xml'],  # Use saved state for positions/velocities
        'equilibrated_pdb': restart_info['structure_pdb'],
        'output_pdb': str(continued_dir / 'final_structure.pdb'),
        'output_state_xml': str(continued_dir / 'final_state.xml'),
        'main_log_file': str(continued_dir / 'continued_main.log'),
        'target_temp': args.target_temp,
        'pressure': args.pressure,
        'timestep': args.timestep,
        'friction_coeff': args.friction_coeff,
        'md_steps': steps_to_run,
        'save_trajectories': args.save_trajectories,
        'save_checkpoints': args.save_checkpoints,
        'trajectory_interval': args.trajectory_interval,
        'checkpoint_interval': args.checkpoint_interval,
        'state_interval': args.state_interval if args.state_interval else args.checkpoint_interval * 2,
    }
    
    # Run the continuation
    results = production_runner.run_npt_production(**run_params)
    
    # Save results
    results_file = continued_dir / 'continued_results.json'
    save_results_to_json(results, str(results_file), run_options=run_params)
    
    # Print summary
    print("\n" + "="*80)
    print("CONTINUATION SUMMARY")
    print("="*80)
    
    if results['success']:
        print(f"Success: Continued simulation completed")
        print(f"Original simulation: {restart_info['simulation_time_ns']:.1f} ns")
        print(f"Additional time: {additional_time_ns:.1f} ns")
        print(f"Total simulation time: {restart_info['simulation_time_ns'] + additional_time_ns:.1f} ns")
        print(f"Output directory: {continued_dir}")
        print(f"Results: {results_file}")
    else:
        print(f"Failure: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("="*80)


def main():
    """Main function to parse arguments and run production MD."""
    parser = argparse.ArgumentParser(
        description=f"OpenMM NPT Production MD Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Regular run
  python %(prog)s system.xml state.xml equilibrated.pdb output.pdb
  
  # Restart from checkpoint
  python %(prog)s --restart checkpoints/simulation_restart_info.json
  
  # Custom checkpoint intervals
  python %(prog)s system.xml state.xml equilibrated.pdb output.pdb --checkpoint-interval 250000 --state-interval 500000
  
  # Long production run with frequent checkpoints
  python %(prog)s system.xml state.xml equilibrated.pdb output.pdb --md-steps 25000000 --checkpoint-interval 500000
"""
    )

    parser.add_argument("system_xml", nargs="?", help="Input system XML file from equilibration")
    parser.add_argument("state_xml", nargs="?", help="Input state XML file from equilibration")
    parser.add_argument("equilibrated_pdb", nargs="?", help="Input equilibrated PDB file for topology")
    parser.add_argument("output_pdb", nargs="?", help="Output final PDB file path")
    
    # Restart functionality
    parser.add_argument("--restart", type=str, help="Restart from checkpoint using restart info JSON file")
    parser.add_argument("--continue-steps", type=int, default=DEFAULT_MD_STEPS,
                       help="Number of additional steps to run when restarting (default: same as normal run)")
    
    # Configuration
    parser.add_argument("--json-config", type=str, help="JSON configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # MD parameters
    parser.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP.value_in_unit(unit.kelvin),
                       help=f"Target temperature in Kelvin (default: {DEFAULT_TARGET_TEMP.value_in_unit(unit.kelvin)})")
    parser.add_argument("--pressure", type=float, default=DEFAULT_PRESSURE.value_in_unit(unit.bar),
                       help=f"Pressure in bar (default: {DEFAULT_PRESSURE.value_in_unit(unit.bar)})")
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds),
                       help=f"Integration timestep in femtoseconds (default: {DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds)})")
    parser.add_argument("--friction-coeff", type=float, default=DEFAULT_FRICTION_COEFF.value_in_unit(unit.picosecond**-1),
                       help=f"Langevin friction coefficient in ps^-1 (default: {DEFAULT_FRICTION_COEFF.value_in_unit(unit.picosecond**-1)})")
    parser.add_argument("--md-steps", type=int, default=DEFAULT_MD_STEPS,
                       help=f"Number of MD steps to run (default: {DEFAULT_MD_STEPS} = 10 ns)")
    
    # Output control
    parser.add_argument("--save-trajectories", default=True, action=argparse.BooleanOptionalAction,
                       help="Save DCD trajectory. Use --no-save-trajectories to disable.")
    parser.add_argument("--save-checkpoints", default=True, action=argparse.BooleanOptionalAction,
                       help="Save checkpoint files. Use --no-save-checkpoints to disable.")
    parser.add_argument("--trajectory-interval", type=int, default=DEFAULT_TRAJECTORY_INTERVAL,
                       help=f"Interval (in steps) for saving trajectory frames (default: {DEFAULT_TRAJECTORY_INTERVAL})")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                       help=f"Interval (in steps) for saving checkpoints (default: {DEFAULT_CHECKPOINT_INTERVAL})")
    parser.add_argument("--state-interval", type=int, 
                       help="Interval (in steps) for saving restart states (default: 2x checkpoint interval)")
    parser.add_argument("--log-frequency", type=int, default=DEFAULT_LOG_FREQUENCY,
                       help=f"Frequency for logging (default: {DEFAULT_LOG_FREQUENCY})")
    parser.add_argument("--analysis-stride", type=int, default=DEFAULT_ANALYSIS_STRIDE,
                       help=f"Stride for collecting analysis data (default: {DEFAULT_ANALYSIS_STRIDE})")
    
    # File output
    parser.add_argument("--output-json", type=str, help="Save results to a specific JSON file (name only)")
    parser.add_argument("--output-state-xml", type=str, help="Save final state to XML file")
    parser.add_argument("--workflow-name", type=str, default="production", help="Workflow name for directory naming")
    parser.add_argument("-n", "--name", dest="run_name", type=str, help="Run name prefix for directory")
    parser.add_argument("--no-run-directory", action="store_true", help="Don't create timestamped run directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--version", action="version", version=f"%(prog)s v{SCRIPT_VERSION}")

    args = parser.parse_args()

    # Handle restart mode
    if args.restart:
        return handle_restart_mode(args)

    # Load JSON config if provided
    config = {}
    if args.json_config:
        with open(args.json_config, 'r') as f:
            config = json.load(f)

    # CLI arguments override JSON config
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_args)

    # Validate required arguments
    system_xml = config.get("system_xml")
    state_xml = config.get("state_xml")
    equilibrated_pdb = config.get("equilibrated_pdb")
    output_pdb = config.get("output_pdb")
    
    if not all([system_xml, state_xml, equilibrated_pdb, output_pdb]):
        parser.error("system_xml, state_xml, equilibrated_pdb, and output_pdb must be provided via command line or JSON config.")

    # --- Directory and Path Setup ---
    run_dir = None
    if not config.get('no_run_directory', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.get("run_name")
        workflow_name = config.get("workflow_name", "production")
        dir_name = f"{run_name}_openmm_{workflow_name}_{timestamp}" if run_name else f"openmm_{workflow_name}_{timestamp}"

        base_output_dir = config.get('output_dir', DEFAULT_OUTPUT_DIR)
        run_dir = Path(base_output_dir) / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Make output paths relative to the new run directory
        output_pdb = run_dir / Path(output_pdb).name
        json_output_name = Path(config.get('output_json', 'production_results.json')).name
        config['output_json'] = run_dir / json_output_name
        
        # Handle output state XML
        if config.get('output_state_xml'):
            config['output_state_xml'] = run_dir / Path(config['output_state_xml']).name
        else:
            # Default: create output state XML based on output PDB name
            output_stem = Path(output_pdb).stem
            config['output_state_xml'] = run_dir / f"{output_stem}_final_state.xml"

    # Set default state interval if not provided
    if not config.get('state_interval'):
        config['state_interval'] = config.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL) * 2

    # --- Logging Setup ---
    main_log_file = run_dir / f"{config.get('workflow_name', 'production')}_main.log" if run_dir else None
    production_runner = OpenMMProduction(log_level=config.get('log_level', 'INFO'))
    production_runner.setup_logging(log_level=config.get('log_level', 'INFO'), 
                                   log_file=str(main_log_file) if main_log_file else None)

    if run_dir:
        production_runner.logger.info(f"Created run directory: {run_dir}")

    # --- Parameter setup ---
    run_params = {
        'system_xml': system_xml,
        'state_xml': state_xml,
        'equilibrated_pdb': equilibrated_pdb,
        'output_pdb': str(output_pdb),
        'output_state_xml': str(config['output_state_xml']) if config.get('output_state_xml') else None,
        'main_log_file': str(main_log_file) if main_log_file else None,
        'target_temp': config.get('target_temp', DEFAULT_TARGET_TEMP.value_in_unit(unit.kelvin)),
        'pressure': config.get('pressure', DEFAULT_PRESSURE.value_in_unit(unit.bar)),
        'timestep': config.get('timestep', DEFAULT_TIMESTEP.value_in_unit(unit.femtoseconds)),
        'friction_coeff': config.get('friction_coeff', DEFAULT_FRICTION_COEFF.value_in_unit(unit.picosecond**-1)),
        'md_steps': config.get('md_steps', DEFAULT_MD_STEPS),
        'save_trajectories': config.get('save_trajectories', True),
        'save_checkpoints': config.get('save_checkpoints', True),
        'trajectory_interval': config.get('trajectory_interval', DEFAULT_TRAJECTORY_INTERVAL),
        'checkpoint_interval': config.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL),
        'state_interval': config.get('state_interval'),
        'log_frequency': config.get('log_frequency', DEFAULT_LOG_FREQUENCY),
        'analysis_stride': config.get('analysis_stride', DEFAULT_ANALYSIS_STRIDE),
    }

    try:
        # Run the production MD simulation
        results = production_runner.run_npt_production(**run_params)

        # Add run info to results
        results['run_directory'] = str(run_dir) if run_dir else None
        results['run_name'] = config.get('run_name')
        results['workflow_name'] = config.get('workflow_name', 'production')

        # Save results to JSON
        json_output_path = config.get("output_json", "production_results.json")
        save_results_to_json(results, str(json_output_path), run_options=run_params)

        # Print summary
        print("\n" + "="*80)
        print("PRODUCTION MD SUMMARY")
        print("="*80)
        
        if results['success']:
            stats = results['statistics']
            trajectory_stats = results.get('trajectory_analysis', {})
            simulation_time_ns = run_params['md_steps'] * run_params['timestep'] / 1e6
            
            print(f"Success: Production MD completed")
            print(f"Simulation time: {simulation_time_ns:.1f} ns ({run_params['md_steps']} steps)")
            if run_dir:
                print(f"Run Directory: {run_dir}")
            
            print("\n--- Input Files ---")
            print(f"   System XML: {run_params['system_xml']}")
            print(f"   State XML: {run_params['state_xml']}")
            print(f"   Input PDB: {run_params['equilibrated_pdb']}")
            
            print("\n--- Output Files ---")
            print(f"   Final PDB: {results['output_pdb']}")
            if results.get("output_state_xml"):
                print(f"   Final State XML: {results['output_state_xml']}")
            if results.get("trajectory_file"):
                print(f"   Trajectory (DCD): {results['trajectory_file']}")
            if results.get("checkpoint_file"):
                print(f"   Checkpoint Directory: {Path(results['checkpoint_file']).parent}")
            
            print("\n--- Log Files ---")
            log_files = results.get('log_files', {})
            if log_files.get("main"): 
                print(f"   Main Log: {log_files['main']}")
            if log_files.get("production"): 
                print(f"   Production Log: {log_files['production']}")

            print("\n--- Simulation Statistics ---")
            if trajectory_stats.get("temperature"):
                temp_stats = trajectory_stats["temperature"]
                print(f"   Temperature: {temp_stats['mean']:.2f} ± {temp_stats['std']:.2f} K")
                print(f"   Temperature range: {temp_stats['min']:.1f} - {temp_stats['max']:.1f} K")
            
            if trajectory_stats.get("potential_energy"):
                pe_stats = trajectory_stats["potential_energy"]
                print(f"   Potential Energy: {pe_stats['mean']:.1f} ± {pe_stats['std']:.1f} kJ/mol")
            
            if trajectory_stats.get("volume"):
                vol_stats = trajectory_stats["volume"]
                print(f"   Volume: {vol_stats['mean']:.2f} ± {vol_stats['std']:.2f} nm³")
            
            if trajectory_stats.get("density"):
                dens_stats = trajectory_stats["density"]
                print(f"   Density: {dens_stats['mean']:.3f} ± {dens_stats['std']:.3f} g/cm³")
            
            print(f"\n   Results JSON: {json_output_path}")
            
            print("\n--- Restart Information ---")
            if run_dir:
                restart_script = run_dir / "restart_simulation.py"
                restart_info = run_dir / "checkpoints" / f"{Path(output_pdb).stem}_restart_info.json"
                print(f"   Restart script: {restart_script}")
                print(f"   Restart info: {restart_info}")
                print(f"   To continue simulation: python {restart_script}")
                print(f"   Or: python {sys.argv[0]} --restart {restart_info}")
            
            print("\n" + "-"*40)
            print("NOTE: Detailed thermodynamic data can be found in the")
            print("'*_production_state.log' file in the run directory.")
            print("Simulation can be restarted from any checkpoint.")
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