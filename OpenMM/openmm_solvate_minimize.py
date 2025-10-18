#!/usr/bin/env python3
"""
OpenMM Solvation and Minimization Script

This script takes a PDB file containing protein and ligand structures, solvates
the system in a water box, and performs energy minimization using OpenMM.
It supports custom ligand template files in XML format for proper parameterization,
and can combine separate ligand PDB files with the protein structure.

The script follows the project coding standards and provides JSON input/output
capabilities for integration with the custom job scheduler.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Script version for tracking and reproducibility
SCRIPT_VERSION = "2.2.0"

# Try to import required libraries
try:
    import openmm
    from openmm import app, unit, System
    from openmm.app import PDBFile, Modeller, ForceField
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

# Default simulation parameters
DEFAULT_WATER_MODEL = "tip3p"
DEFAULT_FORCEFIELD = "amber14-all.xml"
DEFAULT_SOLVENT_FORCEFIELD = "amber14/tip3pfb.xml"
DEFAULT_BOX_PADDING = 1.2  # nm
DEFAULT_IONIC_STRENGTH = 0.1  # M
DEFAULT_MINIMIZATION_STEPS = 50000  # Increased for convergence-based minimization
DEFAULT_MINIMIZATION_TOLERANCE = 1.0  # kJ/mol/nm - More stringent force tolerance
DEFAULT_ENERGY_TOLERANCE = 0.1  # kJ/mol - Energy change tolerance
DEFAULT_CONVERGENCE_WINDOW = 50  # Steps to check for energy convergence
DEFAULT_MIN_STEPS = 100  # Minimum steps before checking convergence
DEFAULT_TEMP_FILE_PREFIX = "openmm_temp_"

# Supported water models and their corresponding forcefields
WATER_MODELS = {
    "tip3p": "amber14/tip3pfb.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "tip5p": "amber14/tip5p.xml",
    "spce": "amber14/spce.xml"
}

# Standard protein forcefields
PROTEIN_FORCEFIELDS = {
    "amber14": "amber14-all.xml",
    "amber99": "amber99sb.xml",
    "charmm36": "charmm36.xml",
    "none": None  
}

# Standard amino acid and common residue names
STANDARD_RESIDUES = {
    # Standard amino acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Modified amino acids
    'MSE', 'SEC', 'PYL',
    # Water molecules
    'HOH', 'WAT', 'H2O',
    # Common ions
    'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+', 'NA', 'CL'
}


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'value_in_unit'):
            # Handle OpenMM quantities
            try:
                return float(obj.value_in_unit(obj.unit))
            except:
                return str(obj)
        return super().default(obj)


def sanitize_for_json(data):
    """
    Recursively sanitize data structure for JSON serialization.
    Converts NumPy types and other non-serializable objects to JSON-compatible types.
    """
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, Path):
        return str(data)
    elif hasattr(data, 'value_in_unit'):
        # Handle OpenMM quantities
        try:
            return float(data.value_in_unit(data.unit))
        except:
            return str(data)
    elif isinstance(data, set):
        return list(data)
    else:
        return data


class LigandTemplateValidator:
    """
    A class to validate and analyze ligand template XML files.
    
    This class provides methods to validate XML format, check for required
    sections, and extract information about ligand templates.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the validator with a logger."""
        self.logger = logger
    
    def validate_xml_file(self, xml_file: str) -> Dict[str, any]:
        """
        Validate an XML ligand template file.
        
        Args:
            xml_file: Path to XML file to validate
            
        Returns:
            Dictionary with validation results and extracted information
        """
        validation_result = {
            "is_valid": False,
            "has_residue_templates": False,
            "has_atom_types": False,
            "has_force_parameters": False,
            "residue_names": [],
            "atom_count": 0,
            "warnings": [],
            "errors": []
        }
        
        try:
            if not os.path.exists(xml_file):
                validation_result["errors"].append(f"File not found: {xml_file}")
                return validation_result
            
            # Parse XML
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                self.logger.info(f"Successfully parsed XML file: {xml_file}")
            except ET.ParseError as e:
                validation_result["errors"].append(f"XML parse error: {e}")
                return validation_result
            
            # Check for ForceField root element
            if root.tag != "ForceField":
                validation_result["warnings"].append(
                    f"Root element is '{root.tag}', expected 'ForceField'"
                )
            
            # Check for residue templates
            residue_templates = root.findall(".//Residue")
            if residue_templates:
                validation_result["has_residue_templates"] = True
                for residue in residue_templates:
                    residue_name = residue.get("name", "UNKNOWN")
                    validation_result["residue_names"].append(residue_name)
                    
                    # Count atoms in this residue
                    atoms = residue.findall("Atom")
                    validation_result["atom_count"] += len(atoms)
                    
                    self.logger.debug(f"Found residue template: {residue_name} with {len(atoms)} atoms")
            
            # Check for atom types
            atom_types = root.findall(".//AtomTypes/Type") + root.findall(".//Type")
            if atom_types:
                validation_result["has_atom_types"] = True
                self.logger.debug(f"Found {len(atom_types)} atom type definitions")
            
            # Check for force parameters
            force_sections = [
                ".//HarmonicBondForce",
                ".//HarmonicAngleForce", 
                ".//PeriodicTorsionForce",
                ".//NonbondedForce"
            ]
            
            for section in force_sections:
                elements = root.findall(section)
                if elements:
                    validation_result["has_force_parameters"] = True
                    self.logger.debug(f"Found force parameters: {section}")
            
            # Overall validation
            if (validation_result["has_residue_templates"] and 
                validation_result["atom_count"] > 0):
                validation_result["is_valid"] = True
                self.logger.info(f"XML template validation successful: "
                               f"{len(validation_result['residue_names'])} residue(s), "
                               f"{validation_result['atom_count']} atoms")
            else:
                validation_result["errors"].append(
                    "XML file lacks required residue templates or atoms"
                )
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            self.logger.error(f"Error validating XML template: {e}")
        
        return validation_result
    
    def extract_residue_info(self, xml_file: str) -> List[Dict[str, any]]:
        """
        Extract detailed residue information from XML template.
        
        Args:
            xml_file: Path to XML file
            
        Returns:
            List of dictionaries containing residue information
        """
        residue_info = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for residue in root.findall(".//Residue"):
                residue_data = {
                    "name": residue.get("name", "UNKNOWN"),
                    "atoms": [],
                    "bonds": [],
                    "external_bonds": []
                }
                
                # Extract atoms
                for atom in residue.findall("Atom"):
                    atom_data = {
                        "name": atom.get("name"),
                        "type": atom.get("type"),
                        "charge": float(atom.get("charge", 0.0))
                    }
                    residue_data["atoms"].append(atom_data)
                
                # Extract bonds
                for bond in residue.findall("Bond"):
                    bond_data = {
                        "from": bond.get("from"),
                        "to": bond.get("to")
                    }
                    residue_data["bonds"].append(bond_data)
                
                # Extract external bonds
                for ext_bond in residue.findall("ExternalBond"):
                    residue_data["external_bonds"].append(ext_bond.get("from"))
                
                residue_info.append(residue_data)
                
        except Exception as e:
            self.logger.warning(f"Could not extract residue information: {e}")
        
        return residue_info


class LigandCombiner:
    """
    A class to combine separate ligand PDB files with a protein structure.
    
    This class handles loading multiple PDB files, validating coordinate systems,
    and creating a combined structure for OpenMM processing.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the ligand combiner with a logger."""
        self.logger = logger
    
    def validate_ligand_pdbs(self, ligand_files: List[str]) -> Dict[str, any]:
        """
        Validate ligand PDB files.
        
        Args:
            ligand_files: List of ligand PDB file paths
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid_files": [],
            "invalid_files": [],
            "total_ligand_atoms": 0,
            "ligand_residue_info": [],
            "warnings": [],
            "errors": []
        }
        
        for ligand_file in ligand_files:
            try:
                if not os.path.exists(ligand_file):
                    validation_result["errors"].append(f"Ligand file not found: {ligand_file}")
                    validation_result["invalid_files"].append(ligand_file)
                    continue
                
                # Try to load the PDB file
                try:
                    ligand_pdb = PDBFile(ligand_file)
                    validation_result["valid_files"].append(ligand_file)
                    
                    # Analyze ligand composition
                    ligand_atoms = len(ligand_pdb.positions)
                    validation_result["total_ligand_atoms"] += ligand_atoms
                    
                    # Get residue information
                    residue_info = []
                    for residue in ligand_pdb.topology.residues():
                        res_info = {
                            "name": residue.name,
                            "chain": residue.chain.id,
                            "id": residue.id,
                            "atoms": len(list(residue.atoms())),
                            "source_file": os.path.basename(ligand_file)
                        }
                        residue_info.append(res_info)
                        validation_result["ligand_residue_info"].append(res_info)
                    
                    self.logger.info(f"Validated ligand PDB: {ligand_file}")
                    self.logger.info(f"  Atoms: {ligand_atoms}")
                    self.logger.info(f"  Residues: {len(residue_info)}")
                    for res in residue_info:
                        self.logger.info(f"    {res['name']} (Chain {res['chain']}, {res['atoms']} atoms)")
                    
                except Exception as pdb_error:
                    validation_result["errors"].append(f"Error loading ligand PDB {ligand_file}: {pdb_error}")
                    validation_result["invalid_files"].append(ligand_file)
                    self.logger.error(f"Failed to load ligand PDB {ligand_file}: {pdb_error}")
                
            except Exception as e:
                validation_result["errors"].append(f"Error processing ligand file {ligand_file}: {e}")
                validation_result["invalid_files"].append(ligand_file)
                self.logger.error(f"Error processing ligand file {ligand_file}: {e}")
        
        if validation_result["invalid_files"]:
            self.logger.warning(f"Invalid ligand files: {validation_result['invalid_files']}")
        
        return validation_result
    
    def combine_structures(self, protein_pdb: PDBFile, ligand_files: List[str]) -> Tuple[Optional[PDBFile], Dict[str, any]]:
        """
        Combine protein and ligand structures into a single PDB.
        
        Args:
            protein_pdb: Loaded protein PDB structure
            ligand_files: List of ligand PDB file paths
            
        Returns:
            Tuple of (combined PDB structure, combination statistics)
        """
        try:
            self.logger.info(f"Combining protein with {len(ligand_files)} ligand file(s)")
            
            # Validate ligand files first
            validation_result = self.validate_ligand_pdbs(ligand_files)
            
            if validation_result["invalid_files"]:
                self.logger.error(f"Cannot proceed with invalid ligand files: {validation_result['invalid_files']}")
                return None, validation_result
            
            if not validation_result["valid_files"]:
                self.logger.warning("No valid ligand files to combine - returning original protein structure")
                return protein_pdb, validation_result
            
            # Start with protein structure
            combined_modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
            
            combination_stats = {
                "original_protein_atoms": len(protein_pdb.positions),
                "ligands_added": 0,
                "total_ligand_atoms": 0,
                "final_atom_count": 0,
                "ligand_details": []
            }
            
            # Add each ligand
            for ligand_file in validation_result["valid_files"]:
                try:
                    self.logger.info(f"Adding ligand from: {ligand_file}")
                    ligand_pdb = PDBFile(ligand_file)
                    
                    # Check for coordinate system compatibility (basic validation)
                    self._validate_coordinate_compatibility(protein_pdb, ligand_pdb, ligand_file)
                    
                    # Add ligand to the combined structure
                    combined_modeller.add(ligand_pdb.topology, ligand_pdb.positions)
                    
                    ligand_atoms = len(ligand_pdb.positions)
                    combination_stats["ligands_added"] += 1
                    combination_stats["total_ligand_atoms"] += ligand_atoms
                    
                    # Get ligand details
                    ligand_detail = {
                        "file": os.path.basename(ligand_file),
                        "atoms": ligand_atoms,
                        "residues": []
                    }
                    
                    for residue in ligand_pdb.topology.residues():
                        ligand_detail["residues"].append({
                            "name": residue.name,
                            "chain": residue.chain.id,
                            "atoms": len(list(residue.atoms()))
                        })
                    
                    combination_stats["ligand_details"].append(ligand_detail)
                    
                    self.logger.info(f"  Added {ligand_atoms} atoms from {os.path.basename(ligand_file)}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add ligand from {ligand_file}: {e}")
                    validation_result["errors"].append(f"Failed to combine ligand {ligand_file}: {e}")
                    continue
            
            combination_stats["final_atom_count"] = len(combined_modeller.positions)
            
            # Create a temporary combined PDB structure
            combined_pdb = self._create_combined_pdb_object(combined_modeller)
            
            # Update validation result with combination stats
            validation_result.update(combination_stats)
            
            self.logger.info(f"Structure combination completed:")
            self.logger.info(f"  Original protein atoms: {combination_stats['original_protein_atoms']}")
            self.logger.info(f"  Ligands added: {combination_stats['ligands_added']}")
            self.logger.info(f"  Total ligand atoms: {combination_stats['total_ligand_atoms']}")
            self.logger.info(f"  Final atom count: {combination_stats['final_atom_count']}")
            
            return combined_pdb, validation_result
            
        except Exception as e:
            self.logger.error(f"Error combining structures: {e}")
            return None, {"errors": [f"Structure combination failed: {e}"]}
    
    def _validate_coordinate_compatibility(self, protein_pdb: PDBFile, ligand_pdb: PDBFile, ligand_file: str) -> None:
        """
        Perform basic validation that ligand coordinates are compatible with protein.
        
        Args:
            protein_pdb: Protein PDB structure
            ligand_pdb: Ligand PDB structure
            ligand_file: Path to ligand file (for error reporting)
        """
        try:
            # Get coordinate ranges
            protein_positions = protein_pdb.positions.value_in_unit(unit.angstroms)
            ligand_positions = ligand_pdb.positions.value_in_unit(unit.angstroms)
            
            protein_min = np.min(protein_positions, axis=0)
            protein_max = np.max(protein_positions, axis=0)
            ligand_min = np.min(ligand_positions, axis=0)
            ligand_max = np.max(ligand_positions, axis=0)
            
            protein_center = (protein_min + protein_max) / 2
            ligand_center = (ligand_min + ligand_max) / 2
            
            # Calculate distance between centers
            center_distance = np.linalg.norm(protein_center - ligand_center)
            
            # Calculate typical protein size
            protein_size = np.linalg.norm(protein_max - protein_min)
            
            self.logger.debug(f"Coordinate compatibility check for {ligand_file}:")
            self.logger.debug(f"  Protein center: [{protein_center[0]:.2f}, {protein_center[1]:.2f}, {protein_center[2]:.2f}] Å")
            self.logger.debug(f"  Ligand center: [{ligand_center[0]:.2f}, {ligand_center[1]:.2f}, {ligand_center[2]:.2f}] Å")
            self.logger.debug(f"  Center distance: {center_distance:.2f} Å")
            self.logger.debug(f"  Protein size: {protein_size:.2f} Å")
            
            # Warning if ligand is very far from protein
            if center_distance > protein_size * 2:
                self.logger.warning(f"Ligand in {ligand_file} is far from protein center ({center_distance:.2f} Å)")
                self.logger.warning("  Please verify that ligand and protein are in the same coordinate system")
            
            # Check for obviously wrong coordinates (e.g., very large values)
            max_reasonable_coord = 1000.0  # Angstroms
            if (np.any(np.abs(ligand_positions) > max_reasonable_coord)):
                self.logger.warning(f"Ligand in {ligand_file} has very large coordinates (> {max_reasonable_coord} Å)")
                self.logger.warning("  This may indicate a coordinate system mismatch")
            
        except Exception as e:
            self.logger.warning(f"Could not validate coordinate compatibility for {ligand_file}: {e}")
    
    def _create_combined_pdb_object(self, modeller: Modeller) -> PDBFile:
        """
        Create a PDBFile object from a Modeller object.
        
        Args:
            modeller: Modeller object with combined structure
            
        Returns:
            PDBFile object
        """
        # Create a temporary PDB-like object
        class CombinedPDB:
            def __init__(self, topology, positions):
                self.topology = topology
                self.positions = positions
        
        return CombinedPDB(modeller.topology, modeller.positions)
    
    def save_combined_structure(self, combined_pdb, output_file: str) -> bool:
        """
        Save the combined structure to a PDB file.
        
        Args:
            combined_pdb: Combined PDB structure
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Saving combined structure to: {output_file}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                PDBFile.writeFile(combined_pdb.topology, combined_pdb.positions, f)
            
            self.logger.info("Combined structure saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving combined structure: {e}")
            return False


class OpenMMSolvator:
    """
    A class to solvate and minimize protein-ligand systems using OpenMM.
    
    This class handles the complete workflow of loading a PDB structure,
    combining with separate ligand structures, adding solvent, applying 
    forcefields (including custom ligand templates), and performing energy 
    minimization.
    """
    
    def __init__(self, log_level: str = "INFO", output_dir: str = DEFAULT_OUTPUT_DIR,
                 create_run_directory: bool = True, workflow_name: str = "solvation",
                 run_name: str = None):
        """
        Initialize the OpenMM Solvator.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            output_dir: Base output directory 
            create_run_directory: Whether to create timestamped run directory
            workflow_name: Name of workflow for directory naming
            run_name: Optional name prefix for the run directory
        """
        self.workflow_name = workflow_name
        self.run_name = run_name
        
        # Create timestamped run directory if requested
        if create_run_directory:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name:
                dir_name = f"{run_name}_openmm_{workflow_name}_{timestamp}"
            else:
                dir_name = f"openmm_{workflow_name}_{timestamp}"
            self.run_dir = Path(output_dir) / dir_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = self.run_dir  # Use run directory as output directory
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir = self.output_dir
        
        # Set up logging first (before creating logger)
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Log directory creation
        if create_run_directory:
            self.logger.info(f"Run directory created: {self.run_dir}")
        else:
            self.logger.info(f"Using output directory: {self.output_dir}")
        
        # Check required dependencies
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required but not installed. Please install it: conda install -c conda-forge openmm")
        
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required but not installed. Please install it: conda install -c conda-forge numpy")
        
        # Initialize helper classes
        self.template_validator = LigandTemplateValidator(self.logger)
        self.ligand_combiner = LigandCombiner(self.logger)
        
        # Statistics tracking
        self.stats = {
            "total_atoms": 0,
            "protein_atoms": 0,
            "ligand_atoms": 0,
            "water_molecules": 0,
            "ion_atoms": 0,
            "box_dimensions": None,
            "initial_energy": None,
            "final_energy": None,
            "minimization_steps_performed": 0,
            "convergence_achieved": False,
            "forcefield_applied": None,
            "water_model_used": None,
            "ligand_templates_used": [],
            "ligand_template_validation": {},
            "ligand_pdbs_used": [],
            "ligand_combination_info": {}
        }
        
        # Temporary files to clean up
        self._temp_files = []
    
    def setup_logging(self, log_level: str) -> None:
        """Set up logging configuration with both console and file output."""
        # Create log filename with timestamp in the run directory
        log_file = self.run_dir / f"{self.workflow_name}.log"
        
        # Configure logging with both file and console handlers
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Store log file path for later reference
        self.log_file_path = str(log_file)
        
        # Log initial message to verify file logging works
        logger.info(f"Logging initialized - console and file output enabled")
        logger.info(f"Log file: {log_file}")
    
    def _add_temp_file(self, filepath: str) -> None:
        """Add a file to the cleanup list."""
        self._temp_files.append(filepath)
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Could not clean up temporary file {temp_file}: {e}")
        self._temp_files.clear()
    
    def validate_inputs(self, pdb_file: str, ligand_templates: List[str] = None, 
                       ligand_pdbs: List[str] = None) -> bool:
        """
        Validate input files exist and are readable.
        
        Args:
            pdb_file: Path to input PDB file
            ligand_templates: List of paths to ligand template XML files
            ligand_pdbs: List of paths to ligand PDB files
            
        Returns:
            True if all inputs are valid, False otherwise
        """
        try:
            if not os.path.exists(pdb_file):
                self.logger.error(f"PDB file not found: {pdb_file}")
                return False
            
            if not os.access(pdb_file, os.R_OK):
                self.logger.error(f"PDB file is not readable: {pdb_file}")
                return False
            
            # Validate ligand templates
            if ligand_templates:
                for template_file in ligand_templates:
                    if not os.path.exists(template_file):
                        self.logger.error(f"Ligand template file not found: {template_file}")
                        return False
                    
                    if not os.access(template_file, os.R_OK):
                        self.logger.error(f"Ligand template file is not readable: {template_file}")
                        return False
                    
                    # Validate XML format and content
                    validation_result = self.template_validator.validate_xml_file(template_file)
                    self.stats["ligand_template_validation"][template_file] = validation_result
                    
                    if not validation_result["is_valid"]:
                        self.logger.error(f"Invalid ligand template: {template_file}")
                        for error in validation_result["errors"]:
                            self.logger.error(f"  - {error}")
                        return False
                    
                    # Log warnings but don't fail validation
                    for warning in validation_result["warnings"]:
                        self.logger.warning(f"Template {template_file}: {warning}")
                    
                    self.logger.info(f"Validated ligand template: {template_file}")
                    self.logger.info(f"  - Residues: {validation_result['residue_names']}")
                    self.logger.info(f"  - Total atoms: {validation_result['atom_count']}")
            
            # Validate ligand PDB files
            if ligand_pdbs:
                for ligand_pdb in ligand_pdbs:
                    if not os.path.exists(ligand_pdb):
                        self.logger.error(f"Ligand PDB file not found: {ligand_pdb}")
                        return False
                    
                    if not os.access(ligand_pdb, os.R_OK):
                        self.logger.error(f"Ligand PDB file is not readable: {ligand_pdb}")
                        return False
                
                # Validate ligand PDB files using the ligand combiner
                ligand_validation = self.ligand_combiner.validate_ligand_pdbs(ligand_pdbs)
                self.stats["ligand_combination_info"] = ligand_validation
                
                if ligand_validation["invalid_files"]:
                    self.logger.error(f"Invalid ligand PDB files found: {ligand_validation['invalid_files']}")
                    for error in ligand_validation["errors"]:
                        self.logger.error(f"  - {error}")
                    return False
                
                self.logger.info(f"Validated {len(ligand_validation['valid_files'])} ligand PDB file(s)")
                for ligand_info in ligand_validation["ligand_residue_info"]:
                    self.logger.info(f"  - {ligand_info['name']} from {ligand_info['source_file']} "
                                   f"({ligand_info['atoms']} atoms)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs: {e}")
            return False
    
    def load_structure(self, pdb_file: str, ligand_pdbs: List[str] = None) -> Tuple[Optional[app.PDBFile], Dict[str, int]]:
        """
        Load PDB structure, optionally combine with ligand PDBs, and analyze composition.
        
        Args:
            pdb_file: Path to input PDB file (usually protein)
            ligand_pdbs: List of paths to ligand PDB files to combine
            
        Returns:
            Tuple of (PDBFile object, composition statistics)
        """
        try:
            self.logger.info(f"Loading PDB structure: {pdb_file}")
            pdb = PDBFile(pdb_file)
            
            # If ligand PDBs are provided, combine them with the protein
            if ligand_pdbs:
                self.logger.info(f"Combining with {len(ligand_pdbs)} ligand PDB file(s)")
                self.stats["ligand_pdbs_used"] = ligand_pdbs
                
                combined_pdb, combination_info = self.ligand_combiner.combine_structures(pdb, ligand_pdbs)
                
                if combined_pdb is None:
                    self.logger.error("Failed to combine protein and ligand structures")
                    return None, {}
                
                # Update stats with combination info
                self.stats["ligand_combination_info"].update(combination_info)
                
                # Save combined structure to a temporary file for reference
                combined_pdb_file = self.output_dir / "combined_input_structure.pdb"
                if self.ligand_combiner.save_combined_structure(combined_pdb, combined_pdb_file):
                    self.logger.info(f"Combined structure saved to: {combined_pdb_file}")
                    self._add_temp_file(str(combined_pdb_file))
                
                # Use the combined structure
                pdb = combined_pdb
            
            # Analyze structure composition
            composition = {
                "total_atoms": len(pdb.positions),
                "total_residues": pdb.topology.getNumResidues(),
                "protein_residues": 0,
                "ligand_residues": 0,
                "water_molecules": 0,
                "chains": pdb.topology.getNumChains(),
                "residue_names": []  # Use list instead of set for JSON serialization
            }
            
            # Analyze residues
            ligand_residues = []
            residue_names_set = set()  # Use set for tracking, then convert to list
            for residue in pdb.topology.residues():
                resname = residue.name.strip()
                residue_names_set.add(resname)
                
                if resname in STANDARD_RESIDUES:
                    if resname in ['HOH', 'WAT', 'H2O']:
                        composition["water_molecules"] += 1
                    elif resname in ['Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+', 'NA', 'CL']:
                        # Count ions separately but don't add to protein residues
                        pass
                    else:
                        composition["protein_residues"] += 1
                else:
                    composition["ligand_residues"] += 1
                    ligand_residues.append({
                        "name": resname,
                        "chain": residue.chain.id,
                        "id": residue.id,
                        "atoms": len(list(residue.atoms()))
                    })
            
            # Convert set to sorted list for JSON serialization
            composition["residue_names"] = sorted(list(residue_names_set))
            
            self.stats.update({
                "total_atoms": composition["total_atoms"],
                "protein_atoms": composition["protein_residues"] * 7,  # Rough estimate
                "ligand_atoms": composition["total_atoms"] - (composition["protein_residues"] * 7)
            })
            
            self.logger.info(f"Structure loaded successfully:")
            self.logger.info(f"  Total atoms: {composition['total_atoms']}")
            self.logger.info(f"  Protein residues: {composition['protein_residues']}")
            self.logger.info(f"  Ligand residues: {composition['ligand_residues']}")
            self.logger.info(f"  Water molecules: {composition['water_molecules']}")
            self.logger.info(f"  Chains: {composition['chains']}")
            
            if ligand_residues:
                self.logger.info("Detected ligand residues:")
                for lig in ligand_residues:
                    self.logger.info(f"  - {lig['name']} (Chain {lig['chain']}, "
                                   f"ID {lig['id']}, {lig['atoms']} atoms)")
            
            # Log ligand combination info if applicable
            if ligand_pdbs and "ligands_added" in self.stats["ligand_combination_info"]:
                combo_info = self.stats["ligand_combination_info"]
                self.logger.info(f"Ligand combination summary:")
                self.logger.info(f"  Original protein atoms: {combo_info.get('original_protein_atoms', 'N/A')}")
                self.logger.info(f"  Ligands added: {combo_info.get('ligands_added', 0)}")
                self.logger.info(f"  Total ligand atoms added: {combo_info.get('total_ligand_atoms', 0)}")
            
            return pdb, composition
            
        except Exception as e:
            self.logger.error(f"Error loading PDB structure: {e}")
            return None, {}
    
    def setup_forcefield(self, protein_ff: str = "amber14", water_model: str = "tip3p",
                        ligand_templates: List[str] = None) -> Optional[ForceField]:
        """
        Set up the forcefield for the system, including ligand templates.
        
        Args:
            protein_ff: Protein forcefield to use (or "none" for no protein forcefield)
            water_model: Water model to use
            ligand_templates: List of ligand template XML files
            
        Returns:
            Configured ForceField object or None if failed
        """
        try:
            self.logger.info(f"Setting up forcefield: {protein_ff} with {water_model} water")
            
            # Validate protein forcefield and water model
            if protein_ff not in PROTEIN_FORCEFIELDS:
                raise ValueError(f"Unknown protein forcefield: {protein_ff}. "
                               f"Available: {list(PROTEIN_FORCEFIELDS.keys())}")
            
            if water_model not in WATER_MODELS:
                raise ValueError(f"Unknown water model: {water_model}. "
                               f"Available: {list(WATER_MODELS.keys())}")
            
            # Build list of forcefield files
            forcefield_files = []
            
            # Add protein forcefield only if not "none"
            if protein_ff != "none":
                forcefield_files.append(PROTEIN_FORCEFIELDS[protein_ff])
            else:
                self.logger.info("No protein forcefield specified - using only ligand templates and water model")
            
            # Add water model
            forcefield_files.append(WATER_MODELS[water_model])
            
            # Add ligand templates if provided
            if ligand_templates:
                for template_file in ligand_templates:
                    forcefield_files.append(template_file)
                    self.stats["ligand_templates_used"].append(template_file)
                    self.logger.info(f"Added ligand template: {template_file}")
                    
                    # Extract and log residue information
                    residue_info = self.template_validator.extract_residue_info(template_file)
                    for residue in residue_info:
                        self.logger.info(f"  Template residue: {residue['name']} "
                                       f"({len(residue['atoms'])} atoms, "
                                       f"{len(residue['bonds'])} bonds)")
            elif protein_ff == "none":
                self.logger.warning("No protein forcefield and no ligand templates specified. "
                                  "This may cause errors if non-water/ion residues are present.")
            
            self.logger.info(f"Creating forcefield with files: {forcefield_files}")
            
            # Create forcefield
            forcefield = ForceField(*forcefield_files)
            
            self.stats.update({
                "forcefield_applied": protein_ff,
                "water_model_used": water_model
            })
            
            self.logger.info("Forcefield setup completed successfully")
            return forcefield
            
        except Exception as e:
            self.logger.error(f"Error setting up forcefield: {e}")
            return None
    
    def solvate_system(self, pdb: app.PDBFile, forcefield: ForceField,
                      box_padding: float = DEFAULT_BOX_PADDING,
                      ionic_strength: float = DEFAULT_IONIC_STRENGTH) -> Optional[Tuple[Modeller, System]]:
        """
        Solvate the system with water and ions.
        
        Args:
            pdb: PDB structure to solvate
            forcefield: Configured forcefield (with ligand templates)
            box_padding: Padding around solute in nm
            ionic_strength: Ionic strength in M
            
        Returns:
            Tuple of (Modeller object, System object) or None if failed
        """
        try:
            self.logger.info(f"Solvating system with {box_padding} nm padding and {ionic_strength} M ionic strength")
            
            # Create modeller
            modeller = Modeller(pdb.topology, pdb.positions)
            
            # Add solvent
            self.logger.info("Adding solvent box...")
            modeller.addSolvent(
                forcefield,
                padding=box_padding * unit.nanometers,
                ionicStrength=ionic_strength * unit.molar
            )
            
            # Get box dimensions
            box_vectors = modeller.topology.getPeriodicBoxVectors()
            if box_vectors:
                box_dims = [vec.value_in_unit(unit.nanometers) for vec in box_vectors]
                self.stats["box_dimensions"] = {
                    "x": box_dims[0][0],
                    "y": box_dims[1][1], 
                    "z": box_dims[2][2]
                }
                self.logger.info(f"Box dimensions: {self.stats['box_dimensions']} nm")
            
            # Count water molecules and ions
            water_count = 0
            ion_count = 0
            residue_counts = {}
            
            for residue in modeller.topology.residues():
                resname = residue.name.strip()
                residue_counts[resname] = residue_counts.get(resname, 0) + 1
                
                if resname in ['HOH', 'WAT', 'H2O']:
                    water_count += 1
                elif resname in ['Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+']:
                    ion_count += 1
            
            self.stats.update({
                "water_molecules": water_count,
                "ion_atoms": ion_count,
                "total_atoms": len(modeller.positions)
            })
            
            self.logger.info(f"Solvation completed:")
            self.logger.info(f"  Total atoms: {self.stats['total_atoms']}")
            self.logger.info(f"  Water molecules: {water_count}")
            self.logger.info(f"  Ion atoms: {ion_count}")
            
            # Log residue composition
            self.logger.debug("Final system composition:")
            for resname, count in sorted(residue_counts.items()):
                self.logger.debug(f"  {resname}: {count}")
            
            # Create system with proper error handling
            self.logger.info("Creating OpenMM system...")
            try:
                system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=app.PME,
                    nonbondedCutoff=1.0 * unit.nanometers,
                    constraints=app.HBonds
                )
                self.logger.info("OpenMM system created successfully")
            except Exception as system_error:
                self.logger.error(f"Failed to create OpenMM system: {system_error}")
                self.logger.error("This may indicate missing parameters for some residues")
                
                # Check if ligand templates cover all non-standard residues
                self._diagnose_missing_parameters(modeller.topology, forcefield)
                raise
            
            return modeller, system
            
        except Exception as e:
            self.logger.error(f"Error solvating system: {e}")
            return None
    
    def _diagnose_missing_parameters(self, topology, forcefield) -> None:
        """
        Diagnose missing parameters in the system.
        
        Args:
            topology: OpenMM Topology object
            forcefield: ForceField object
        """
        try:
            self.logger.info("Diagnosing missing parameters...")
            
            # Get template residues from ligand files
            template_residues = set()
            for template_file in self.stats["ligand_templates_used"]:
                validation_result = self.stats["ligand_template_validation"].get(template_file, {})
                template_residues.update(validation_result.get("residue_names", []))
            
            # Find residues without parameters
            missing_residues = set()
            for residue in topology.residues():
                resname = residue.name.strip()
                if resname not in STANDARD_RESIDUES and resname not in template_residues:
                    missing_residues.add(resname)
            
            if missing_residues:
                self.logger.error(f"Residues without parameters: {sorted(missing_residues)}")
                self.logger.error("Consider creating ligand templates for these residues")
                
                # If ligand PDBs were used, suggest creating templates for them
                if self.stats["ligand_pdbs_used"]:
                    ligand_info = self.stats.get("ligand_combination_info", {})
                    ligand_residues_from_pdbs = set()
                    for res_info in ligand_info.get("ligand_residue_info", []):
                        ligand_residues_from_pdbs.add(res_info["name"])
                    
                    missing_from_ligand_pdbs = missing_residues.intersection(ligand_residues_from_pdbs)
                    if missing_from_ligand_pdbs:
                        self.logger.error(f"Missing templates for ligand PDB residues: {sorted(missing_from_ligand_pdbs)}")
                        self.logger.error("You need to provide ligand templates (--ligand-templates) for these residues")
            else:
                self.logger.info("All residues appear to have parameter coverage")
                
        except Exception as e:
            self.logger.warning(f"Could not diagnose missing parameters: {e}")
    
    def minimize_energy(self, modeller: Modeller, system: System,
                       max_steps: int = DEFAULT_MINIMIZATION_STEPS,
                       force_tolerance: float = DEFAULT_MINIMIZATION_TOLERANCE,
                       energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
                       convergence_window: int = DEFAULT_CONVERGENCE_WINDOW,
                       min_steps: int = DEFAULT_MIN_STEPS,
                       minimize_until_converged: bool = True) -> Tuple[bool, Dict[str, float]]:
        """
        Perform energy minimization on the solvated system with proper convergence criteria.
        
        Args:
            modeller: Solvated system modeller
            system: OpenMM system object
            max_steps: Maximum minimization steps (safety limit)
            force_tolerance: Force convergence tolerance in kJ/mol/nm
            energy_tolerance: Energy change tolerance in kJ/mol over convergence_window steps
            convergence_window: Number of steps to check for energy convergence
            min_steps: Minimum steps before checking convergence
            minimize_until_converged: If True, continue until converged (up to max_steps)
            
        Returns:
            Tuple of (success, energy_info)
        """
        try:
            self.logger.info(f"Starting energy minimization:")
            self.logger.info(f"  Maximum steps: {max_steps}")
            self.logger.info(f"  Force tolerance: {force_tolerance} kJ/mol/nm")
            self.logger.info(f"  Energy tolerance: {energy_tolerance} kJ/mol over {convergence_window} steps")
            self.logger.info(f"  Minimum steps: {min_steps}")
            self.logger.info(f"  Continue until converged: {minimize_until_converged}")
            
            # Create integrator and simulation
            integrator = openmm.LangevinIntegrator(
                300 * unit.kelvin,
                1.0 / unit.picoseconds,
                2.0 * unit.femtoseconds
            )
            
            simulation = app.Simulation(modeller.topology, system, integrator)
            simulation.context.setPositions(modeller.positions)
            
            # Get initial energy and forces
            initial_state = simulation.context.getState(getEnergy=True, getForces=True)
            initial_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            initial_forces = initial_state.getForces(asNumpy=True)
            initial_max_force = np.max(np.sqrt(np.sum(initial_forces**2, axis=1)))
            initial_rms_force = np.sqrt(np.mean(np.sum(initial_forces**2, axis=1)))
            
            self.stats["initial_energy"] = initial_energy
            
            self.logger.info(f"Initial state:")
            self.logger.info(f"  Potential energy: {initial_energy:.2f} kJ/mol")
            self.logger.info(f"  Maximum force: {initial_max_force:.2f} kJ/mol/nm")
            self.logger.info(f"  RMS force: {initial_rms_force:.2f} kJ/mol/nm")
            
            # Energy and convergence tracking
            energy_history = []
            force_history = []
            step_count = 0
            converged = False
            convergence_reason = "Maximum steps reached"
            
            if minimize_until_converged:
                # Iterative minimization with convergence checking
                batch_size = min(100, max_steps // 10)  # Minimize in batches
                self.logger.info(f"Running iterative minimization in batches of {batch_size} steps...")
                
                while step_count < max_steps and not converged:
                    # Run a batch of minimization steps
                    remaining_steps = min(batch_size, max_steps - step_count)
                    
                    self.logger.debug(f"Minimization batch: steps {step_count + 1}-{step_count + remaining_steps}")
                    simulation.minimizeEnergy(
                        tolerance=force_tolerance * unit.kilojoules_per_mole / unit.nanometers,
                        maxIterations=remaining_steps
                    )
                    step_count += remaining_steps
                    
                    # Get current state
                    current_state = simulation.context.getState(getEnergy=True, getForces=True)
                    current_energy = current_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    current_forces = current_state.getForces(asNumpy=True)
                    current_max_force = np.max(np.sqrt(np.sum(current_forces**2, axis=1)))
                    current_rms_force = np.sqrt(np.mean(np.sum(current_forces**2, axis=1)))
                    
                    # Store history
                    energy_history.append(current_energy)
                    force_history.append(current_max_force)
                    
                    # Check convergence criteria after minimum steps
                    if step_count >= min_steps:
                        # Force-based convergence - EXPLICIT CONVERSION TO PYTHON BOOL
                        force_converged = bool(current_max_force < force_tolerance)
                        
                        # Energy-based convergence (check if we have enough history)
                        energy_converged = False
                        if len(energy_history) >= convergence_window:
                            energy_window = energy_history[-convergence_window:]
                            energy_change = abs(energy_window[-1] - energy_window[0])
                            energy_converged = bool(energy_change < energy_tolerance)
                        
                        # Combined convergence check
                        if force_converged and energy_converged:
                            converged = True
                            convergence_reason = "Force and energy convergence achieved"
                        elif force_converged:
                            # If force converged but energy hasn't, check if energy is stable
                            if len(energy_history) >= convergence_window:
                                recent_energies = energy_history[-convergence_window:]
                                energy_std = np.std(recent_energies)
                                if energy_std < energy_tolerance:
                                    converged = True
                                    convergence_reason = "Force convergence and energy stability achieved"
                    
                    # Progress logging every few batches
                    if step_count % (batch_size * 5) == 0 or converged:
                        self.logger.info(f"Step {step_count}: Energy = {current_energy:.2f} kJ/mol, "
                                       f"Max Force = {current_max_force:.3f} kJ/mol/nm, "
                                       f"RMS Force = {current_rms_force:.3f} kJ/mol/nm")
                        
                        if step_count >= min_steps and len(energy_history) >= convergence_window:
                            energy_change = abs(energy_history[-1] - energy_history[-convergence_window])
                            self.logger.debug(f"Energy change over last {convergence_window} steps: {energy_change:.4f} kJ/mol")
            
            else:
                # Traditional fixed-step minimization
                self.logger.info("Running fixed-step minimization...")
                simulation.minimizeEnergy(
                    tolerance=force_tolerance * unit.kilojoules_per_mole / unit.nanometers,
                    maxIterations=max_steps
                )
                step_count = max_steps  # OpenMM doesn't report actual steps taken
                convergence_reason = "Fixed-step minimization completed"
            
            # Get final state
            final_state = simulation.context.getState(getEnergy=True, getForces=True)
            final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            final_forces = final_state.getForces(asNumpy=True)
            final_max_force = np.max(np.sqrt(np.sum(final_forces**2, axis=1)))
            final_rms_force = np.sqrt(np.mean(np.sum(final_forces**2, axis=1)))
            
            self.stats["final_energy"] = final_energy
            energy_change = final_energy - initial_energy
            
            # Determine final convergence status - EXPLICIT CONVERSION TO PYTHON BOOL
            force_converged = bool(final_max_force < force_tolerance)
            if minimize_until_converged and len(energy_history) >= convergence_window:
                energy_window = energy_history[-convergence_window:]
                energy_change_final = abs(energy_window[-1] - energy_window[0])
                energy_converged = bool(energy_change_final < energy_tolerance)
                final_convergence = bool(converged or (force_converged and energy_converged))
            else:
                energy_converged = True  # Assume converged for fixed-step mode
                final_convergence = force_converged
            
            self.stats.update({
                "minimization_steps_performed": step_count,
                "convergence_achieved": final_convergence
            })
            
            # Comprehensive logging of results
            self.logger.info(f"Energy minimization completed:")
            self.logger.info(f"  Steps performed: {step_count}")
            self.logger.info(f"  Convergence reason: {convergence_reason}")
            self.logger.info(f"  Final convergence status: {'CONVERGED' if final_convergence else 'NOT CONVERGED'}")
            self.logger.info(f"  Initial energy: {initial_energy:.2f} kJ/mol")
            self.logger.info(f"  Final energy: {final_energy:.2f} kJ/mol")
            self.logger.info(f"  Energy change: {energy_change:.2f} kJ/mol")
            self.logger.info(f"  Initial max force: {initial_max_force:.3f} kJ/mol/nm")
            self.logger.info(f"  Final max force: {final_max_force:.3f} kJ/mol/nm")
            self.logger.info(f"  Initial RMS force: {initial_rms_force:.3f} kJ/mol/nm")
            self.logger.info(f"  Final RMS force: {final_rms_force:.3f} kJ/mol/nm")
            self.logger.info(f"  Force converged: {'YES' if force_converged else 'NO'} (< {force_tolerance} kJ/mol/nm)")
            if minimize_until_converged and len(energy_history) >= convergence_window:
                self.logger.info(f"  Energy converged: {'YES' if energy_converged else 'NO'} (< {energy_tolerance} kJ/mol over {convergence_window} steps)")
            
            # Update modeller with minimized positions
            minimized_positions = simulation.context.getState(getPositions=True).getPositions()
            modeller.positions = minimized_positions
            
            energy_info = {
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_change": energy_change,
                "initial_max_force": initial_max_force,
                "final_max_force": final_max_force,
                "initial_rms_force": initial_rms_force,
                "final_rms_force": final_rms_force,
                "steps_performed": step_count,
                "convergence_achieved": final_convergence,
                "convergence_reason": convergence_reason,
                "force_converged": force_converged,
                "energy_converged": energy_converged if minimize_until_converged else True,
                "force_tolerance_used": force_tolerance,
                "energy_tolerance_used": energy_tolerance,
                "convergence_window_used": convergence_window,
                "minimize_until_converged": minimize_until_converged
            }
            
            return True, energy_info
            
        except Exception as e:
            self.logger.error(f"Error during energy minimization: {e}")
            return False, {}
    
    def save_minimized_structure(self, modeller: Modeller, output_file: str) -> bool:
        """
        Save the minimized structure to a PDB file.
        
        Args:
            modeller: Minimized system modeller
            output_file: Path to output PDB file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output file is in the run directory if using relative path
            if not os.path.isabs(output_file):
                output_file = self.output_dir / output_file
            else:
                output_file = Path(output_file)
            
            os.makedirs(output_file.parent, exist_ok=True)
            
            self.logger.info(f"Saving minimized structure: {output_file}")
            
            with open(output_file, 'w') as f:
                PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
                            
            self.logger.info("Minimized structure saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving minimized structure: {e}")
            return False
    
    def run_solvation_minimization(self, pdb_file: str, output_file: str,
                                  ligand_templates: List[str] = None,
                                  ligand_pdbs: List[str] = None,
                                  protein_ff: str = "amber14",
                                  water_model: str = "tip3p",
                                  box_padding: float = DEFAULT_BOX_PADDING,
                                  ionic_strength: float = DEFAULT_IONIC_STRENGTH,
                                  max_steps: int = DEFAULT_MINIMIZATION_STEPS,
                                  force_tolerance: float = DEFAULT_MINIMIZATION_TOLERANCE,
                                  energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
                                  convergence_window: int = DEFAULT_CONVERGENCE_WINDOW,
                                  min_steps: int = DEFAULT_MIN_STEPS,
                                  minimize_until_converged: bool = True) -> Dict[str, any]:
        """
        Run the complete solvation and minimization workflow.
        
        Args:
            pdb_file: Input PDB file path (usually protein)
            output_file: Output minimized PDB file path
            ligand_templates: List of ligand template XML files
            ligand_pdbs: List of ligand PDB files to combine with protein
            protein_ff: Protein forcefield to use
            water_model: Water model to use
            box_padding: Box padding in nm
            ionic_strength: Ionic strength in M
            max_steps: Maximum minimization steps
            force_tolerance: Force convergence tolerance in kJ/mol/nm
            energy_tolerance: Energy change tolerance in kJ/mol
            convergence_window: Steps to check for energy convergence
            min_steps: Minimum steps before checking convergence
            minimize_until_converged: Continue until converged (up to max_steps)
            
        Returns:
            Dictionary with processing results and statistics
        """
        self.logger.info(f"Starting solvation and minimization workflow")
        self.logger.info(f"Input PDB: {pdb_file}")
        self.logger.info(f"Output PDB: {output_file}")
        if ligand_templates:
            self.logger.info(f"Ligand templates: {ligand_templates}")
        if ligand_pdbs:
            self.logger.info(f"Ligand PDBs: {ligand_pdbs}")
        
        # Reset statistics
        self.stats = {key: 0 if isinstance(val, (int, float)) else [] if isinstance(val, list) else {} if isinstance(val, dict) else None
                     for key, val in self.stats.items()}
        
        try:
            # Validate inputs
            if not self.validate_inputs(pdb_file, ligand_templates, ligand_pdbs):
                raise ValueError("Input validation failed")
            
            # Load structure (with optional ligand combination)
            pdb, composition = self.load_structure(pdb_file, ligand_pdbs)
            if pdb is None:
                raise RuntimeError("Failed to load PDB structure")
            
            # Setup forcefield with ligand templates
            forcefield = self.setup_forcefield(protein_ff, water_model, ligand_templates)
            if forcefield is None:
                raise RuntimeError("Failed to setup forcefield")
            
            # Solvate system
            solvation_result = self.solvate_system(pdb, forcefield, box_padding, ionic_strength)
            if solvation_result is None:
                raise RuntimeError("Failed to solvate system")
            
            modeller, system = solvation_result
            
            # Minimize energy
            minimization_success, energy_info = self.minimize_energy(
                modeller, system, max_steps, force_tolerance, energy_tolerance,
                convergence_window, min_steps, minimize_until_converged
            )
            if not minimization_success:
                raise RuntimeError("Energy minimization failed")
            
            # Save minimized structure
            save_success = self.save_minimized_structure(modeller, output_file)
            if not save_success:
                raise RuntimeError("Failed to save minimized structure")
            
            # Prepare results
            results = {
                "success": True,
                "input_file": pdb_file,
                "output_file": str(output_file),
                "run_directory": str(self.run_dir),
                "log_file": getattr(self, 'log_file_path', None),
                "workflow_name": self.workflow_name,
                "run_name": self.run_name,
                "ligand_templates": ligand_templates or [],
                "ligand_pdbs": ligand_pdbs or [],
                "parameters": {
                    "protein_forcefield": protein_ff,
                    "water_model": water_model,
                    "box_padding_nm": box_padding,
                    "ionic_strength_M": ionic_strength,
                    "max_minimization_steps": max_steps,
                    "force_tolerance": force_tolerance,
                    "energy_tolerance": energy_tolerance,
                    "convergence_window": convergence_window,
                    "min_steps": min_steps,
                    "minimize_until_converged": minimize_until_converged
                },
                "composition": composition,
                "energy_info": energy_info,
                "statistics": self.stats.copy()
            }
            
            self.logger.info("Solvation and minimization workflow completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            results = {
                "success": False,
                "error": str(e),
                "input_file": pdb_file,
                "output_file": str(output_file),
                "run_directory": str(self.run_dir),
                "log_file": getattr(self, 'log_file_path', None),
                "workflow_name": self.workflow_name,
                "ligand_templates": ligand_templates or [],
                "ligand_pdbs": ligand_pdbs or [],
                "statistics": self.stats.copy()
            }
            return results
        
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()


def create_directories() -> None:
    """Create input and output directories if they don't exist."""
    for directory in [DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)


def load_config_from_json(json_file: str) -> Dict[str, any]:
    """Load configuration from JSON file."""
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ["input_pdb", "output_pdb"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing from JSON config")
        
        return config
    except Exception as e:
        logging.error(f"Error loading JSON configuration: {e}")
        raise


def save_results_to_json(results: Dict[str, any], output_file: str,
                        run_options: Dict[str, any] = None) -> None:
    """Save processing results to JSON file with metadata and proper serialization."""
    try:
        # Create comprehensive output structure
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "script_version": SCRIPT_VERSION,
                "openmm_available": OPENMM_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE
            },
            "run_options": run_options or {},
            "results": results,
            "summary": {
                "processing_successful": results.get("success", False),
                "input_file": results.get("input_file"),
                "output_file": results.get("output_file"),
                "run_directory": results.get("run_directory"),
                "log_file": results.get("log_file"),
                "workflow_name": results.get("workflow_name"),
                "run_name": results.get("run_name"),
                "ligand_templates_used": len(results.get("ligand_templates", [])),
                "ligand_template_files": results.get("ligand_templates", []),
                "ligand_pdbs_used": len(results.get("ligand_pdbs", [])),
                "ligand_pdb_files": results.get("ligand_pdbs", []),
                "total_atoms_final": results.get("statistics", {}).get("total_atoms", 0),
                "water_molecules_added": results.get("statistics", {}).get("water_molecules", 0),
                "energy_change": results.get("energy_info", {}).get("energy_change"),
                "convergence_achieved": results.get("statistics", {}).get("convergence_achieved", False),
                "template_validation_summary": {},
                "ligand_combination_summary": {}
            }
        }
        
        # Add template validation summary
        stats = results.get("statistics", {})
        template_validation = stats.get("ligand_template_validation", {})
        for template_file, validation in template_validation.items():
            filename = Path(template_file).name
            output_data["summary"]["template_validation_summary"][filename] = {
                "valid": validation.get("is_valid", False),
                "residue_names": validation.get("residue_names", []),
                "atom_count": validation.get("atom_count", 0),
                "has_force_parameters": validation.get("has_force_parameters", False)
            }
        
        # Add ligand combination summary
        ligand_combination = stats.get("ligand_combination_info", {})
        if ligand_combination:
            output_data["summary"]["ligand_combination_summary"] = {
                "ligands_added": ligand_combination.get("ligands_added", 0),
                "total_ligand_atoms": ligand_combination.get("total_ligand_atoms", 0),
                "original_protein_atoms": ligand_combination.get("original_protein_atoms", 0),
                "ligand_files": [detail.get("file", "") for detail in ligand_combination.get("ligand_details", [])]
            }
        
        # Sanitize the entire data structure before serialization
        output_data = sanitize_for_json(output_data)
        
        # Ensure output file is in the run directory if available
        if "run_directory" in results and not os.path.isabs(output_file):
            output_file = Path(results["run_directory"]) / output_file
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write JSON with custom encoder as fallback
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, sort_keys=True, ensure_ascii=False)
        except TypeError as e:
            # Fallback to custom encoder if standard serialization fails
            logging.warning(f"Standard JSON serialization failed ({e}), using custom encoder...")
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, sort_keys=True, 
                         cls=CustomJSONEncoder, ensure_ascii=False)
        
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")
        # Try to save a minimal version with error info
        try:
            minimal_data = {
                "error": "Failed to save complete results",
                "error_details": str(e),
                "success": results.get("success", False),
                "input_file": str(results.get("input_file", "")),
                "output_file": str(results.get("output_file", "")),
                "timestamp": datetime.now().isoformat()
            }
            with open(output_file, 'w') as f:
                json.dump(minimal_data, f, indent=2)
            logging.info(f"Minimal results saved to: {output_file}")
        except Exception as minimal_error:
            logging.error(f"Failed to save even minimal results to JSON: {minimal_error}")
        raise


def collect_run_options(args) -> Dict[str, any]:
    """Collect all run options from args into a dictionary."""
    return {
        "script_version": SCRIPT_VERSION,
        "input_pdb": getattr(args, 'input_pdb', None),
        "output_pdb": getattr(args, 'output_pdb', None),
        "ligand_templates": getattr(args, 'ligand_templates', []),
        "ligand_pdbs": getattr(args, 'ligand_pdbs', []),
        "json_config_file": getattr(args, 'json_config', None),
        "log_level": args.log_level,
        "protein_forcefield": args.protein_ff,
        "water_model": args.water_model,
        "box_padding": args.box_padding,
        "ionic_strength": args.ionic_strength,
        "minimization_steps": args.max_steps,
        "force_tolerance": getattr(args, 'force_tolerance', DEFAULT_MINIMIZATION_TOLERANCE),
        "energy_tolerance": getattr(args, 'energy_tolerance', DEFAULT_ENERGY_TOLERANCE),
        "convergence_window": getattr(args, 'convergence_window', DEFAULT_CONVERGENCE_WINDOW),
        "min_steps": getattr(args, 'min_steps', DEFAULT_MIN_STEPS),
        "minimize_until_converged": not getattr(args, 'no_converge', False),
        "create_run_directory": getattr(args, 'create_run_directory', True),
        "workflow_name": getattr(args, 'workflow_name', 'solvation'),
        "run_name": getattr(args, 'run_name', None),
        "command_line": " ".join(sys.argv)
    }


def parse_ligand_templates(template_arg: str) -> List[str]:
    """
    Parse ligand template argument which can be a single file or comma-separated list.
    
    Args:
        template_arg: String containing template file path(s)
        
    Returns:
        List of template file paths
    """
    if not template_arg:
        return []
    
    # Split by comma and clean up whitespace
    templates = [t.strip() for t in template_arg.split(',') if t.strip()]
    
    # Expand any glob patterns
    expanded_templates = []
    for template in templates:
        if '*' in template or '?' in template:
            # Handle glob patterns
            from glob import glob
            matches = glob(template)
            if matches:
                expanded_templates.extend(matches)
            else:
                logging.warning(f"No files found matching pattern: {template}")
        else:
            expanded_templates.append(template)
    
    return expanded_templates


def parse_ligand_pdbs(ligand_arg: str) -> List[str]:
    """
    Parse ligand PDB argument which can be a single file or comma-separated list.
    
    Args:
        ligand_arg: String containing ligand PDB file path(s)
        
    Returns:
        List of ligand PDB file paths
    """
    if not ligand_arg:
        return []
    
    # Split by comma and clean up whitespace
    ligands = [l.strip() for l in ligand_arg.split(',') if l.strip()]
    
    # Expand any glob patterns
    expanded_ligands = []
    for ligand in ligands:
        if '*' in ligand or '?' in ligand:
            # Handle glob patterns
            from glob import glob
            matches = glob(ligand)
            if matches:
                expanded_ligands.extend(matches)
            else:
                logging.warning(f"No files found matching pattern: {ligand}")
        else:
            expanded_ligands.append(ligand)
    
    return expanded_ligands


def main():
    """Main function to handle command-line arguments and execute OpenMM workflow."""
    parser = argparse.ArgumentParser(
        description=f"OpenMM Solvation and Minimization Script (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s input/complex.pdb output/minimized.pdb
  python %(prog)s input.pdb output.pdb --ligand-templates ligand.xml
  python %(prog)s input.pdb output.pdb --ligand-templates "lig1.xml,lig2.xml"
  python %(prog)s input.pdb output.pdb --ligand-templates "templates/*.xml"
  python %(prog)s protein.pdb output.pdb --ligand ligand.pdb --ligand-templates ligand.xml
  python %(prog)s protein.pdb output.pdb --ligand "lig1.pdb,lig2.pdb" --ligand-templates "lig1.xml,lig2.xml"
  python %(prog)s protein.pdb output.pdb --ligand "ligands/*.pdb" --ligand-templates "templates/*.xml"
  python %(prog)s --json-config config.json
  python %(prog)s input.pdb output.pdb --protein-ff charmm36 --water-model tip4pew
  python %(prog)s input.pdb output.pdb --box-padding 1.5 --ionic-strength 0.15 --max-steps 5000
  python %(prog)s input.pdb output.pdb --workflow-name "protein_prep" --no-run-directory
  python %(prog)s input.pdb output.pdb -n "experiment1" --workflow-name "solvation"
  python %(prog)s protein.pdb output.pdb -n "complex_test" --ligand ligand.pdb --ligand-templates ligand.xml

Supported protein forcefields: """ + ", ".join(PROTEIN_FORCEFIELDS.keys()) + """
Supported water models: """ + ", ".join(WATER_MODELS.keys()) + """

Ligand Integration:
  - Use --ligand to specify separate ligand PDB files to combine with protein
  - Ligand PDBs must be in the same coordinate system as the protein
  - Specify single ligand: --ligand ligand.pdb
  - Multiple ligands: --ligand "lig1.pdb,lig2.pdb"
  - Use wildcards: --ligand "ligands/*.pdb"
  - Combined structure is saved as "combined_input_structure.pdb" in run directory

Ligand Templates:
  - Specify single template: --ligand-templates template.xml
  - Multiple templates: --ligand-templates "template1.xml,template2.xml"
  - Use wildcards: --ligand-templates "templates/*.xml"
  - Templates should contain residue definitions for ligands in your PDB
  - Required when using separate ligand PDB files with non-standard residues

Convergence Criteria:
  - By default, minimizes until both force and energy convergence are achieved
  - Force convergence: maximum force on any atom < force_tolerance
  - Energy convergence: energy change < energy_tolerance over convergence_window steps
  - Use --no-converge for traditional fixed-step minimization
  - Convergence is checked only after min_steps have been performed

Output Organization:
  - By default, creates timestamped run directories like: output/openmm_solvation_20241212_143052/
  - Use -n/--name to add prefix: output/experiment1_openmm_solvation_20241212_143052/
  - Use --no-run-directory to save directly to specified output paths
  - Use --workflow-name to customize the directory name
  - All logs and results are organized in the run directory
"""
    )
    
    parser.add_argument("input_pdb", nargs="?", help="Input PDB file path (protein structure)")
    parser.add_argument("output_pdb", nargs="?", help="Output minimized PDB file path")
    parser.add_argument("--ligand", type=str, dest="ligand_pdbs",
                       help="Ligand PDB file(s) to combine with protein - single file, comma-separated list, or glob pattern")
    parser.add_argument("--ligand-templates", type=str, 
                       help="Ligand template XML file(s) - single file, comma-separated list, or glob pattern")
    parser.add_argument("--json-config", type=str, help="JSON configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level (default: INFO)")
    parser.add_argument("--protein-ff", choices=list(PROTEIN_FORCEFIELDS.keys()),
                       default="amber14", help="Protein forcefield (default: amber14)")
    parser.add_argument("--water-model", choices=list(WATER_MODELS.keys()),
                       default="tip3p", help="Water model (default: tip3p)")
    parser.add_argument("--box-padding", type=float, default=DEFAULT_BOX_PADDING,
                       help=f"Box padding in nm (default: {DEFAULT_BOX_PADDING})")
    parser.add_argument("--ionic-strength", type=float, default=DEFAULT_IONIC_STRENGTH,
                       help=f"Ionic strength in M (default: {DEFAULT_IONIC_STRENGTH})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MINIMIZATION_STEPS,
                       help=f"Maximum minimization steps (default: {DEFAULT_MINIMIZATION_STEPS})")
    parser.add_argument("--force-tolerance", type=float, default=DEFAULT_MINIMIZATION_TOLERANCE,
                       help=f"Force convergence tolerance in kJ/mol/nm (default: {DEFAULT_MINIMIZATION_TOLERANCE})")
    parser.add_argument("--energy-tolerance", type=float, default=DEFAULT_ENERGY_TOLERANCE,
                       help=f"Energy change tolerance in kJ/mol (default: {DEFAULT_ENERGY_TOLERANCE})")
    parser.add_argument("--convergence-window", type=int, default=DEFAULT_CONVERGENCE_WINDOW,
                       help=f"Steps to check for energy convergence (default: {DEFAULT_CONVERGENCE_WINDOW})")
    parser.add_argument("--min-steps", type=int, default=DEFAULT_MIN_STEPS,
                       help=f"Minimum steps before checking convergence (default: {DEFAULT_MIN_STEPS})")
    parser.add_argument("--no-converge", action="store_true",
                       help="Use fixed-step minimization instead of converging until criteria met")
    parser.add_argument("--output-json", type=str,
                       help="Save results to a specific JSON file")
    parser.add_argument("--workflow-name", type=str, default="solvation",
                       help="Workflow name for directory naming (default: solvation)")
    parser.add_argument("-n", "--name", dest="run_name", type=str,
                       help="Run name prefix for directory (creates: {name}_openmm_{workflow}_{timestamp})")
    parser.add_argument("--no-run-directory", action="store_true",
                       help="Don't create timestamped run directory, use direct paths")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--version", action="version",
                       version=f"OpenMM Solvation and Minimization Script v{SCRIPT_VERSION}")
    
    args = parser.parse_args()
    
    create_directories()
    
    # Handle run directory creation option
    args.create_run_directory = not args.no_run_directory
    
    run_options = collect_run_options(args)
    
    try:
        if args.json_config:
            config = load_config_from_json(args.json_config)
            input_pdb = config.get("input_pdb")
            output_pdb = config.get("output_pdb")
            ligand_templates_str = config.get("ligand_templates", args.ligand_templates)
            ligand_pdbs_str = config.get("ligand_pdbs", args.ligand_pdbs)
            protein_ff = config.get("protein_forcefield", args.protein_ff)
            water_model = config.get("water_model", args.water_model)
            box_padding = config.get("box_padding", args.box_padding)
            ionic_strength = config.get("ionic_strength", args.ionic_strength)
            max_steps = config.get("minimization_steps", args.max_steps)
            force_tolerance = config.get("force_tolerance", getattr(args, 'force_tolerance', DEFAULT_MINIMIZATION_TOLERANCE))
            energy_tolerance = config.get("energy_tolerance", getattr(args, 'energy_tolerance', DEFAULT_ENERGY_TOLERANCE))
            convergence_window = config.get("convergence_window", getattr(args, 'convergence_window', DEFAULT_CONVERGENCE_WINDOW))
            min_steps = config.get("min_steps", getattr(args, 'min_steps', DEFAULT_MIN_STEPS))
            minimize_until_converged = config.get("minimize_until_converged", not getattr(args, 'no_converge', False))
            output_json = config.get("output_json", args.output_json)
            workflow_name = config.get("workflow_name", args.workflow_name)
            run_name = config.get("run_name", args.run_name)
            create_run_directory = config.get("create_run_directory", args.create_run_directory)
            
            run_options.update({
                "config_source": "JSON file",
                "input_pdb": input_pdb,
                "output_pdb": output_pdb,
                "ligand_templates": ligand_templates_str,
                "ligand_pdbs": ligand_pdbs_str,
                "protein_forcefield": protein_ff,
                "water_model": water_model,
                "box_padding": box_padding,
                "ionic_strength": ionic_strength,
                "minimization_steps": max_steps,
                "force_tolerance": force_tolerance,
                "energy_tolerance": energy_tolerance,
                "convergence_window": convergence_window,
                "min_steps": min_steps,
                "minimize_until_converged": minimize_until_converged,
                "workflow_name": workflow_name,
                "run_name": run_name,
                "create_run_directory": create_run_directory
            })
        else:
            if not args.input_pdb or not args.output_pdb:
                parser.error("Either provide input_pdb and output_pdb arguments, or use --json-config")
            input_pdb = args.input_pdb
            output_pdb = args.output_pdb
            ligand_templates_str = args.ligand_templates
            ligand_pdbs_str = args.ligand_pdbs
            protein_ff = args.protein_ff
            water_model = args.water_model
            box_padding = args.box_padding
            ionic_strength = args.ionic_strength
            max_steps = args.max_steps
            force_tolerance = args.force_tolerance
            energy_tolerance = args.energy_tolerance
            convergence_window = args.convergence_window
            min_steps = args.min_steps
            minimize_until_converged = not args.no_converge
            output_json = args.output_json
            workflow_name = args.workflow_name
            run_name = args.run_name
            create_run_directory = args.create_run_directory
            run_options["config_source"] = "Command line arguments"
        
        # Parse ligand templates and ligand PDBs
        ligand_templates = parse_ligand_templates(ligand_templates_str) if ligand_templates_str else []
        ligand_pdbs = parse_ligand_pdbs(ligand_pdbs_str) if ligand_pdbs_str else []
        
        run_options.update({
            "ligand_templates": ligand_templates,
            "ligand_pdbs": ligand_pdbs
        })
        
        if not os.path.exists(input_pdb):
            raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")
        
        # Validate ligand template files exist
        for template_file in ligand_templates:
            if not os.path.exists(template_file):
                raise FileNotFoundError(f"Ligand template file not found: {template_file}")
        
        # Validate ligand PDB files exist
        for ligand_file in ligand_pdbs:
            if not os.path.exists(ligand_file):
                raise FileNotFoundError(f"Ligand PDB file not found: {ligand_file}")
        
        # Initialize OpenMM solvator with timestamped directory option
        solvator = OpenMMSolvator(
            log_level=args.log_level, 
            output_dir=args.output_dir,
            create_run_directory=create_run_directory,
            workflow_name=workflow_name,
            run_name=run_name
        )
        
        # Run the workflow
        results = solvator.run_solvation_minimization(
            pdb_file=input_pdb,
            output_file=output_pdb,
            ligand_templates=ligand_templates,
            ligand_pdbs=ligand_pdbs,
            protein_ff=protein_ff,
            water_model=water_model,
            box_padding=box_padding,
            ionic_strength=ionic_strength,
            max_steps=max_steps,
            force_tolerance=force_tolerance,
            energy_tolerance=energy_tolerance,
            convergence_window=convergence_window,
            min_steps=min_steps,
            minimize_until_converged=minimize_until_converged
        )
        
        # Save results to JSON
        json_output_path = output_json or "openmm_solvation_results.json"
        save_results_to_json(results, json_output_path, run_options)
        
        # Print final summary
        print("\n" + "="*80)
        print(f"OPENMM SOLVATION AND MINIMIZATION RESULTS (v{SCRIPT_VERSION})")
        print("="*80)
        print(f"Input file: {input_pdb}")
        print(f"Processing successful: {'YES' if results['success'] else 'NO'}")
        
        if results['success']:
            print(f"Output file: {results['output_file']}")
            print(f"Run directory: {results.get('run_directory', 'N/A')}")
            print(f"Log file: {results.get('log_file', 'N/A')}")
            print(f"Workflow name: {results.get('workflow_name', 'N/A')}")
            if results.get('run_name'):
                print(f"Run name: {results.get('run_name', 'N/A')}")
            
            ligand_templates_used = results.get('ligand_templates', [])
            ligand_pdbs_used = results.get('ligand_pdbs', [])
            
            print(f"Ligand templates used: {len(ligand_templates_used)}")
            if ligand_templates_used:
                for i, template in enumerate(ligand_templates_used, 1):
                    template_name = Path(template).name
                    print(f"  {i}. {template_name}")
            
            print(f"Ligand PDBs combined: {len(ligand_pdbs_used)}")
            if ligand_pdbs_used:
                for i, ligand_pdb in enumerate(ligand_pdbs_used, 1):
                    ligand_name = Path(ligand_pdb).name
                    print(f"  {i}. {ligand_name}")
            
            print("-" * 50)
            print("System Parameters:")
            params = results.get('parameters', {})
            print(f"  Protein forcefield: {params.get('protein_forcefield', 'N/A')}")
            print(f"  Water model: {params.get('water_model', 'N/A')}")
            print(f"  Box padding: {params.get('box_padding_nm', 'N/A')} nm")
            print(f"  Ionic strength: {params.get('ionic_strength_M', 'N/A')} M")
            print(f"  Maximum steps: {params.get('max_minimization_steps', 'N/A')}")
            print(f"  Force tolerance: {params.get('force_tolerance', 'N/A')} kJ/mol/nm")
            print(f"  Energy tolerance: {params.get('energy_tolerance', 'N/A')} kJ/mol")
            print(f"  Convergence window: {params.get('convergence_window', 'N/A')} steps")
            print(f"  Minimize until converged: {'YES' if params.get('minimize_until_converged', True) else 'NO'}")
            
            print("-" * 50)
            print("System Composition:")
            stats = results.get('statistics', {})
            comp = results.get('composition', {})
            print(f"  Total atoms (final): {stats.get('total_atoms', 'N/A')}")
            print(f"  Protein residues: {comp.get('protein_residues', 'N/A')}")
            print(f"  Ligand residues: {comp.get('ligand_residues', 'N/A')}")
            print(f"  Water molecules added: {stats.get('water_molecules', 'N/A')}")
            print(f"  Ion atoms added: {stats.get('ion_atoms', 'N/A')}")
            box_dims = stats.get('box_dimensions')
            if box_dims:
                print(f"  Box dimensions: {box_dims['x']:.2f} x {box_dims['y']:.2f} x {box_dims['z']:.2f} nm")
            
            # Show ligand combination info if applicable
            ligand_combination = stats.get('ligand_combination_info', {})
            if ligand_combination and ligand_combination.get('ligands_added', 0) > 0:
                print("-" * 50)
                print("Ligand Combination Summary:")
                print(f"  Original protein atoms: {ligand_combination.get('original_protein_atoms', 'N/A')}")
                print(f"  Ligands added: {ligand_combination.get('ligands_added', 0)}")
                print(f"  Total ligand atoms added: {ligand_combination.get('total_ligand_atoms', 0)}")
                print(f"  Final combined atoms: {ligand_combination.get('final_atom_count', 'N/A')}")
                
                ligand_details = ligand_combination.get('ligand_details', [])
                if ligand_details:
                    print("  Ligand details:")
                    for detail in ligand_details:
                        print(f"    {detail.get('file', 'Unknown')}: {detail.get('atoms', 0)} atoms")
                        for residue in detail.get('residues', []):
                            print(f"      - {residue.get('name', 'UNK')} (Chain {residue.get('chain', '?')}, {residue.get('atoms', 0)} atoms)")
            
            # Show ligand template validation results
            template_validation = stats.get('ligand_template_validation', {})
            if template_validation:
                print("-" * 50)
                print("Ligand Template Validation:")
                for template_file, validation in template_validation.items():
                    template_name = Path(template_file).name
                    print(f"  {template_name}:")
                    print(f"    Valid: {'YES' if validation.get('is_valid') else 'NO'}")
                    print(f"    Residues: {validation.get('residue_names', [])}")
                    print(f"    Atoms: {validation.get('atom_count', 0)}")
                    print(f"    Force parameters: {'YES' if validation.get('has_force_parameters') else 'NO'}")
            
            print("-" * 50)
            print("Energy Information:")
            energy_info = results.get('energy_info', {})
            initial_energy = energy_info.get('initial_energy')
            final_energy = energy_info.get('final_energy')
            energy_change = energy_info.get('energy_change')
            steps_performed = energy_info.get('steps_performed', 'N/A')
            convergence = energy_info.get('convergence_achieved', False)
            convergence_reason = energy_info.get('convergence_reason', 'N/A')
            force_converged = energy_info.get('force_converged', False)
            energy_converged = energy_info.get('energy_converged', False)
            initial_max_force = energy_info.get('initial_max_force')
            final_max_force = energy_info.get('final_max_force')
            
            print(f"  Steps performed: {steps_performed}")
            print(f"  Convergence reason: {convergence_reason}")
            if initial_energy is not None:
                print(f"  Initial energy: {initial_energy:.2f} kJ/mol")
            if final_energy is not None:
                print(f"  Final energy: {final_energy:.2f} kJ/mol")
            if energy_change is not None:
                print(f"  Energy change: {energy_change:.2f} kJ/mol")
            if initial_max_force is not None:
                print(f"  Initial max force: {initial_max_force:.3f} kJ/mol/nm")
            if final_max_force is not None:
                print(f"  Final max force: {final_max_force:.3f} kJ/mol/nm")
            print(f"  Overall convergence: {'YES' if convergence else 'NO'}")
            print(f"  Force convergence: {'YES' if force_converged else 'NO'}")
            print(f"  Energy convergence: {'YES' if energy_converged else 'NO'}")
        else:
            print(f"Error: {results.get('error', 'Unknown error occurred')}")
        
        print("-" * 50)
        print("Output files:")
        if results['success']:
            print(f"  Minimized structure: {results['output_file']}")
            if ligand_pdbs_used:
                combined_structure_path = Path(results.get('run_directory', '.')) / "combined_input_structure.pdb"
                print(f"  Combined input structure: {combined_structure_path}")
        
        # Handle JSON output path
        if "run_directory" in results and not os.path.isabs(json_output_path):
            final_json_path = Path(results["run_directory"]) / json_output_path
        else:
            final_json_path = json_output_path
        print(f"  Results JSON: {final_json_path}")
        
        if results.get('run_directory'):
            print(f"  All files in: {results['run_directory']}")
        
        print("-" * 50)
        print("Dependencies:")
        print(f"  OpenMM: {'AVAILABLE' if OPENMM_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  NumPy: {'AVAILABLE' if NUMPY_AVAILABLE else 'NOT AVAILABLE'}")
        print("="*80)
        
        if not results['success']:
            print("\nERROR: Processing failed. Check the log for details.")
            if ligand_pdbs_used and not ligand_templates_used:
                print("\nHINT: When using separate ligand PDB files, you may need to provide")
                print("      ligand templates (--ligand-templates) for non-standard residues.")
            sys.exit(1)
        else:
            print(f"\nSUCCESS: Solvation and minimization completed successfully!")
            if ligand_pdbs_used:
                print(f"         Successfully combined {len(ligand_pdbs_used)} ligand file(s) with protein!")
        
    except Exception as e:
        logging.error(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()