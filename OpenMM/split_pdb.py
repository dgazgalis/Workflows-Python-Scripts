#!/usr/bin/env python3
"""
PDB Structure Splitter Script - Enhanced Version (FIXED + AUTO LIGAND OUTPUT)

This script takes a prepared PDB file containing both protein and ligand components
and splits them into separate files:
- protein.pdb: Contains the protein structure (optionally with ions)
- Individual ligand SDF files: Automatically created for each ligand

Key improvements:
- NO LONGER REQUIRES ligand_output argument - automatically generates ligand filenames
- Option to include ions with protein (--include-ions)
- Properly handles multiple ligands in a single PDB file
- Creates individual SDF files for ligands automatically
- Better error handling and fallback mechanisms
- Enhanced logging and statistics tracking
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import tempfile

# Script version for tracking and reproducibility
SCRIPT_VERSION = "2.1.0"

# Try to import required libraries
try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython is not available. Please install it: conda install -c conda-forge biopython")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not available. Please install it: conda install -c conda-forge rdkit")

# Configuration constants
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Standard amino acid residue names
STANDARD_AMINO_ACIDS = {
    # Standard amino acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Modified amino acids
    'MSE', 'SEC', 'PYL',
    # Phosphorylated amino acids (common naming conventions)
    'SEP',  # Phosphoserine (most common)
    'TPO',  # Phosphothreonine (most common)
    'PTR',  # Phosphotyrosine (most common)
    'PSR',  
    'TYP',  
    'S1P',  
    'T1P',  
    'Y1P',  
    # Capping groups
    'ACE', 'NME',
}

# Water molecules
WATER_MOLECULES = {
    'HOH', 'WAT', 'H2O', 'TIP', 'SPC'
}

# Common ions
COMMON_IONS = {
    'Na+', 'CL-', 'K+', 'MG', 'MG2', 'CA', 'CA2', 'ZN2', 'FE2', 'FE3',
    'NA', 'CL', 'MN', 'ZN', 'FE', 'CU', 'NI', 'CO'
}

# Buffer components and crystallization additives (usually not of interest as ligands)
BUFFER_COMPONENTS = {
    'SO4', 'PO4', 'NO3', 'ACT', 'EDO', 'GOL', 'PEG', 'TRS', 'HEZ'
}

# Common cofactors that might be considered ligands or excluded
COMMON_COFACTORS = {
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'NAD', 'FAD', 'FMN',
    'COA', 'HEM', 'HEME', 'PLP', 'B12', 'THF'
}


class ProteinSelector(Select):
    """Select protein chains, standard amino acids, and optionally water/ions."""
    
    def __init__(self, include_water: bool = False, include_ions: bool = False):
        """
        Initialize protein selector.
        
        Args:
            include_water: Whether to include water molecules
            include_ions: Whether to include ions
        """
        self.include_water = include_water
        self.include_ions = include_ions
    
    def accept_residue(self, residue):
        """Accept protein residues and optionally water/ions."""
        resname = residue.get_resname().strip()
        
        # Always accept standard amino acids
        if resname in STANDARD_AMINO_ACIDS:
            return True
        
        # Optionally include water
        if self.include_water and resname in WATER_MOLECULES:
            return True
        
        # Optionally include ions
        if self.include_ions and resname in COMMON_IONS:
            return True
        
        return False


class LigandSelector(Select):
    """Select only ligand molecules (HETATM records excluding water/ions/buffers)."""
    
    def __init__(self, exclude_cofactors: bool = False, exclude_buffers: bool = True):
        """
        Initialize ligand selector.
        
        Args:
            exclude_cofactors: Whether to exclude common cofactors from ligands
            exclude_buffers: Whether to exclude buffer components from ligands
        """
        self.exclude_cofactors = exclude_cofactors
        self.exclude_buffers = exclude_buffers
    
    def accept_residue(self, residue):
        """Accept only ligand residues (HETATM excluding water/ions/buffers)."""
        resname = residue.get_resname().strip()
        
        # Exclude standard amino acids
        if resname in STANDARD_AMINO_ACIDS:
            return False
        
        # Exclude water molecules
        if resname in WATER_MOLECULES:
            return False
        
        # Exclude common ions
        if resname in COMMON_IONS:
            return False
        
        # Optionally exclude buffer components
        if self.exclude_buffers and resname in BUFFER_COMPONENTS:
            return False
        
        # Optionally exclude common cofactors
        if self.exclude_cofactors and resname in COMMON_COFACTORS:
            return False
        
        return True


class IndividualLigandSelector(Select):
    """Select only a specific ligand residue by chain and residue ID."""
    
    def __init__(self, target_chain: str, target_resnum: int, target_resname: str, 
                 exclude_cofactors: bool = False, exclude_buffers: bool = True):
        """
        Initialize selector for a specific ligand residue.
        
        Args:
            target_chain: Chain ID of the target ligand
            target_resnum: Residue number of the target ligand
            target_resname: Residue name of the target ligand
            exclude_cofactors: Whether to exclude common cofactors from ligands
            exclude_buffers: Whether to exclude buffer components from ligands
        """
        self.target_chain = target_chain
        self.target_resnum = target_resnum
        self.target_resname = target_resname
        self.exclude_cofactors = exclude_cofactors
        self.exclude_buffers = exclude_buffers
    
    def accept_chain(self, chain):
        """Accept only the target chain."""
        return chain.get_id() == self.target_chain
    
    def accept_residue(self, residue):
        """Accept only the specific target ligand residue."""
        resname = residue.get_resname().strip()
        resnum = residue.get_id()[1]
        
        # Must match target residue exactly
        if (resname != self.target_resname or resnum != self.target_resnum):
            return False
        
        # Apply same filtering logic as LigandSelector
        if resname in STANDARD_AMINO_ACIDS:
            return False
        
        if resname in WATER_MOLECULES:
            return False
        
        if resname in COMMON_IONS:
            return False
        
        if self.exclude_buffers and resname in BUFFER_COMPONENTS:
            return False
        
        if self.exclude_cofactors and resname in COMMON_COFACTORS:
            return False
        
        return True


class PDBSplitter:
    """
    Enhanced PDB splitter that properly handles multiple ligands with automatic output naming.
    """
    
    def __init__(self, log_level: str = "INFO", exclude_cofactors: bool = False,
                 include_water: bool = False, include_ions: bool = False,
                 exclude_buffers: bool = True):
        """
        Initialize the PDBSplitter.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            exclude_cofactors: Whether to exclude common cofactors from ligands
            include_water: Whether to include water molecules in protein output
            include_ions: Whether to include ions in protein output
            exclude_buffers: Whether to exclude buffer components from ligands
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.exclude_cofactors = exclude_cofactors
        self.include_water = include_water
        self.include_ions = include_ions
        self.exclude_buffers = exclude_buffers
        
        # Check required dependencies
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required but not installed. Please install it: conda install -c conda-forge biopython")
        
        # Statistics tracking
        self.stats = {
            "total_residues": 0,
            "protein_residues": 0,
            "ligand_residues": 0,
            "water_molecules": 0,
            "ion_molecules": 0,
            "cofactor_molecules": 0,
            "buffer_molecules": 0,
            "protein_chains": 0,
            "ligand_molecules": 0,
            "unique_ligand_types": 0,
            "rdkit_conversion_success": False,
            "individual_ligands_processed": 0,
            "sdf_molecules_written": 0,
            "sdf_files_created": [],
        }
        
        # Track individual ligands found
        self.ligand_inventory = []
    
    def setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=LOG_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
    
    def identify_ligands(self, structure) -> List[Dict[str, any]]:
        """
        Identify all individual ligand molecules in the structure.
        
        Args:
            structure: BioPython structure object
            
        Returns:
            List of dictionaries containing ligand information
        """
        ligands = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    resnum = residue.get_id()[1]
                    chain_id = chain.get_id()
                    
                    # Check if this is a ligand using same logic as LigandSelector
                    if resname in STANDARD_AMINO_ACIDS:
                        continue
                    if resname in WATER_MOLECULES:
                        continue
                    if resname in COMMON_IONS:
                        continue
                    if self.exclude_buffers and resname in BUFFER_COMPONENTS:
                        continue
                    if self.exclude_cofactors and resname in COMMON_COFACTORS:
                        continue
                    
                    # This is a ligand - add to inventory
                    ligand_info = {
                        'chain': chain_id,
                        'resnum': resnum,
                        'resname': resname,
                        'full_id': f"{chain_id}_{resname}_{resnum}",
                        'atom_count': len(list(residue.get_atoms()))
                    }
                    ligands.append(ligand_info)
        
        self.ligand_inventory = ligands
        self.stats["ligand_molecules"] = len(ligands)
        self.stats["unique_ligand_types"] = len(set(lig['resname'] for lig in ligands))
        
        return ligands
    
    def analyze_pdb_structure(self, structure) -> Dict[str, List[str]]:
        """
        Analyze PDB structure to identify different types of residues.
        
        Args:
            structure: BioPython structure object
            
        Returns:
            Dictionary containing lists of residue names by category
        """
        analysis = {
            "protein_residues": [],
            "ligand_residues": [],
            "water_molecules": [],
            "ions": [],
            "cofactors": [],
            "buffer_components": [],
            "unknown": []
        }
        
        # Reset stats
        for key in self.stats:
            if isinstance(self.stats[key], int):
                self.stats[key] = 0
        self.stats["rdkit_conversion_success"] = False
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    self.stats["total_residues"] += 1
                    
                    if resname in STANDARD_AMINO_ACIDS:
                        analysis["protein_residues"].append(resname)
                        self.stats["protein_residues"] += 1
                    elif resname in WATER_MOLECULES:
                        analysis["water_molecules"].append(resname)
                        self.stats["water_molecules"] += 1
                    elif resname in COMMON_IONS:
                        analysis["ions"].append(resname)
                        self.stats["ion_molecules"] += 1
                    elif resname in BUFFER_COMPONENTS:
                        analysis["buffer_components"].append(resname)
                        self.stats["buffer_molecules"] += 1
                    elif resname in COMMON_COFACTORS:
                        analysis["cofactors"].append(resname)
                        self.stats["cofactor_molecules"] += 1
                    else:
                        analysis["ligand_residues"].append(resname)
                        self.stats["ligand_residues"] += 1
        
        # Count unique chains with protein residues
        protein_chains = set()
        for model in structure:
            for chain in model:
                has_protein = any(res.get_resname().strip() in STANDARD_AMINO_ACIDS 
                                for res in chain)
                if has_protein:
                    protein_chains.add(chain.get_id())
        self.stats["protein_chains"] = len(protein_chains)
        
        return analysis
    
    def generate_ligand_output_path(self, input_file_path: str, base_name: str = None) -> str:
        """
        Generate automatic ligand output path based on input file.
        
        Args:
            input_file_path: Path to input PDB file
            base_name: Optional base name for ligand files
            
        Returns:
            Base path for ligand output files
        """
        input_dir = os.path.dirname(os.path.abspath(input_file_path))
        input_basename = os.path.splitext(os.path.basename(input_file_path))[0]
        
        if base_name:
            ligand_base = os.path.join(input_dir, base_name)
        else:
            ligand_base = os.path.join(input_dir, f"{input_basename}_ligands")
        
        return ligand_base
    
    def normalize_output_path(self, output_file: str, default_extension: str = None, 
                            use_input_dir: bool = False, input_file_path: str = None) -> str:
        """
        Normalize output file path and ensure it has proper directory structure.
        """
        if not output_file:
            raise ValueError("Output file path cannot be empty")
        
        # If use_input_dir is True and we have an input file path, place output in input directory
        if use_input_dir and input_file_path:
            input_dir = os.path.dirname(os.path.abspath(input_file_path))
            # If output_file is just a filename (no directory), place it in input directory
            if not os.path.dirname(output_file):
                output_file = os.path.join(input_dir, output_file)
        
        # Convert to absolute path
        output_file = os.path.abspath(output_file)
        
        # Add default extension if no extension is present and default is provided
        if default_extension and not os.path.splitext(output_file)[1]:
            output_file += default_extension
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        return output_file
    
    def extract_protein(self, structure, output_file: str, input_file_path: str = None) -> bool:
        """
        Extract protein chains and save to PDB file.
        """
        try:
            # Normalize the output path - place in input directory if just filename given
            use_input_dir = not os.path.dirname(output_file)
            output_file = self.normalize_output_path(output_file, ".pdb", use_input_dir, input_file_path)
            
            io = PDBIO()
            io.set_structure(structure)
            
            # Use protein selector with optional water/ion inclusion
            protein_selector = ProteinSelector(
                include_water=self.include_water,
                include_ions=self.include_ions
            )
            io.save(output_file, protein_selector)
            
            self.logger.info(f"Protein structure saved to: {output_file}")
            if self.include_water:
                self.logger.info("  - Water molecules included")
            if self.include_ions:
                self.logger.info("  - Ions included")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting protein: {e}")
            return False
    
    def extract_single_ligand_to_pdb(self, structure, ligand_info: Dict[str, any], 
                                   temp_file: str) -> bool:
        """
        Extract a single ligand to a temporary PDB file.
        """
        try:
            io = PDBIO()
            io.set_structure(structure)
            
            selector = IndividualLigandSelector(
                target_chain=ligand_info['chain'],
                target_resnum=ligand_info['resnum'],
                target_resname=ligand_info['resname'],
                exclude_cofactors=self.exclude_cofactors,
                exclude_buffers=self.exclude_buffers
            )
            
            io.save(temp_file, selector)
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting ligand {ligand_info['full_id']}: {e}")
            return False
    
    def convert_ligands_to_sdf(self, structure, input_file_path: str, base_name: str = None) -> bool:
        """
        Extract all ligands and convert to individual SDF files using RDKit.
        """
        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit not available - cannot convert to SDF format")
            return self._fallback_to_pdb(structure, input_file_path, base_name)
        
        try:
            # Generate automatic ligand output path
            ligand_base = self.generate_ligand_output_path(input_file_path, base_name)
            base_dir = os.path.dirname(ligand_base)
            base_name_final = os.path.basename(ligand_base)
            
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            # Identify all ligands in the structure
            ligands = self.identify_ligands(structure)
            
            if not ligands:
                self.logger.warning("No ligands found in structure")
                return False
            
            self.logger.info(f"Found {len(ligands)} ligand molecules:")
            for lig in ligands:
                self.logger.info(f"  - {lig['full_id']} ({lig['atom_count']} atoms)")
            
            successful_conversions = 0
            created_files = []
            
            # Process each ligand individually
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, ligand_info in enumerate(ligands):
                    temp_pdb = os.path.join(temp_dir, f"ligand_{i}_{ligand_info['full_id']}.pdb")
                    
                    # Extract individual ligand to temporary PDB
                    if not self.extract_single_ligand_to_pdb(structure, ligand_info, temp_pdb):
                        self.logger.warning(f"Failed to extract ligand {ligand_info['full_id']}")
                        continue
                    
                    # Convert PDB to molecule using RDKit
                    try:
                        mol = Chem.MolFromPDBFile(temp_pdb, removeHs=False)
                        
                        if mol is not None:
                            # Add properties to the molecule
                            mol.SetProp("_Name", ligand_info['full_id'])
                            mol.SetProp("Chain", ligand_info['chain'])
                            mol.SetProp("ResidueNumber", str(ligand_info['resnum']))
                            mol.SetProp("ResidueName", ligand_info['resname'])
                            mol.SetProp("AtomCount", str(ligand_info['atom_count']))
                            
                            # Create individual SDF file for this ligand
                            sdf_filename = f"{base_name_final}_{ligand_info['full_id']}.sdf"
                            
                            if base_dir:
                                sdf_path = os.path.join(base_dir, sdf_filename)
                            else:
                                sdf_path = sdf_filename
                            
                            # Write individual SDF file
                            writer = Chem.SDWriter(sdf_path)
                            writer.write(mol)
                            writer.close()
                            
                            created_files.append(sdf_path)
                            successful_conversions += 1
                            self.logger.info(f"Successfully converted ligand {ligand_info['full_id']} to SDF: {sdf_path}")
                        else:
                            self.logger.warning(f"RDKit could not parse ligand {ligand_info['full_id']}")
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing ligand {ligand_info['full_id']}: {e}")
                        continue
            
            # Update statistics
            self.stats["individual_ligands_processed"] = len(ligands)
            self.stats["sdf_molecules_written"] = successful_conversions
            self.stats["sdf_files_created"] = created_files
            
            if successful_conversions > 0:
                self.stats["rdkit_conversion_success"] = True
                self.logger.info(f"Successfully converted {successful_conversions}/{len(ligands)} ligands to individual SDF files")
                return True
            else:
                self.logger.warning("No ligands could be converted to SDF format")
                return self._fallback_to_pdb(structure, input_file_path, base_name)
                
        except Exception as e:
            self.logger.error(f"Error converting ligands to SDF: {e}")
            return self._fallback_to_pdb(structure, input_file_path, base_name)
    
    def _fallback_to_pdb(self, structure, input_file_path: str, base_name: str = None) -> bool:
        """
        Fallback method to save ligands as PDB when SDF conversion fails.
        """
        # Generate automatic ligand output path
        ligand_base = self.generate_ligand_output_path(input_file_path, base_name)
        pdb_output = ligand_base + "_all.pdb"
        
        self.logger.info(f"Falling back to PDB format: {pdb_output}")
        
        try:
            io = PDBIO()
            io.set_structure(structure)
            
            ligand_selector = LigandSelector(
                exclude_cofactors=self.exclude_cofactors,
                exclude_buffers=self.exclude_buffers
            )
            io.save(pdb_output, ligand_selector)
            
            self.logger.info(f"Ligands saved as PDB: {pdb_output}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in PDB fallback: {e}")
            return False
    
    def split_pdb_file(self, input_file: str, protein_output: str, 
                       ligand_base_name: str = None, use_sdf: bool = True) -> Dict[str, any]:
        """
        Split a PDB file into protein and ligand components.
        
        Args:
            input_file: Path to input PDB file
            protein_output: Path to output protein PDB file
            ligand_base_name: Optional base name for ligand files (auto-generated if None)
            use_sdf: Whether to convert ligands to SDF format
            
        Returns:
            Dictionary with processing statistics and results
        """
        self.logger.info(f"Processing PDB file: {input_file}")
        
        try:
            # Parse PDB file
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("input_structure", input_file)
            
            # Analyze structure
            analysis = self.analyze_pdb_structure(structure)
            
            # Log analysis results
            self.logger.info("Structure analysis:")
            for category, residues in analysis.items():
                unique_residues = set(residues)
                if unique_residues:
                    self.logger.info(f"  {category}: {len(residues)} residues "
                                   f"({len(unique_residues)} unique types: {', '.join(sorted(unique_residues))})")
            
            # Extract protein
            protein_success = self.extract_protein(structure, protein_output, input_file)
            
            # Extract ligands with automatic naming
            ligand_success = False
            actual_ligand_format = "None"
            
            if use_sdf and RDKIT_AVAILABLE:
                ligand_success = self.convert_ligands_to_sdf(structure, input_file, ligand_base_name)
                if ligand_success and self.stats["rdkit_conversion_success"]:
                    actual_ligand_format = "SDF"
                else:
                    actual_ligand_format = "PDB"
            else:
                # Use PDB format
                ligand_success = self._fallback_to_pdb(structure, input_file, ligand_base_name)
                actual_ligand_format = "PDB" if ligand_success else "None"
            
            # Prepare results
            results = {
                "success": protein_success and ligand_success,
                "protein_extracted": protein_success,
                "ligand_extracted": ligand_success,
                "ligand_format": actual_ligand_format,
                "analysis": analysis,
                "statistics": self.stats.copy(),
                "ligand_inventory": self.ligand_inventory.copy()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDB file: {e}")
            raise


def create_directories() -> None:
    """Create input and output directories if they don't exist."""
    for directory in [DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)


def load_config_from_json(json_file: str) -> Dict[str, any]:
    """Load configuration from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON configuration: {e}")
        raise


def save_results_to_json(results: Dict[str, any], output_file: str, 
                         run_options: Dict[str, any] = None) -> None:
    """Save processing results to JSON file with metadata."""
    try:
        # Create comprehensive output structure
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "script_version": SCRIPT_VERSION,
                "biopython_available": BIOPYTHON_AVAILABLE,
                "rdkit_available": RDKIT_AVAILABLE
            },
            "run_options": run_options or {},
            "results": results,
            "summary": {
                "processing_successful": results.get("success", False),
                "protein_extracted": results.get("protein_extracted", False),
                "ligand_extracted": results.get("ligand_extracted", False),
                "ligand_format": results.get("ligand_format", "Unknown"),
                "total_residues_processed": results.get("statistics", {}).get("total_residues", 0),
                "protein_residues": results.get("statistics", {}).get("protein_residues", 0),
                "ligand_residues": results.get("statistics", {}).get("ligand_residues", 0),
                "individual_ligands_found": results.get("statistics", {}).get("ligand_molecules", 0),
                "unique_ligand_types": results.get("statistics", {}).get("unique_ligand_types", 0),
                "sdf_molecules_written": results.get("statistics", {}).get("sdf_molecules_written", 0),
                "sdf_files_created": results.get("statistics", {}).get("sdf_files_created", [])
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        logging.info(f"Results saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")
        raise


def collect_run_options(args) -> Dict[str, any]:
    """Collect all run options from args into a dictionary."""
    return {
        "script_version": SCRIPT_VERSION,
        "input_file": getattr(args, 'input_pdb', None),
        "protein_output": getattr(args, 'protein_output', None),
        "ligand_base_name": getattr(args, 'ligand_base_name', None),
        "json_config_file": getattr(args, 'json_config', None),
        "log_level": args.log_level,
        "exclude_cofactors": args.exclude_cofactors,
        "exclude_buffers": getattr(args, 'exclude_buffers', True),
        "include_water": args.include_water,
        "include_ions": getattr(args, 'include_ions', False),
        "use_sdf_format": not args.no_sdf,
        "command_line": " ".join(sys.argv)
    }


def main():
    """Main function to handle command-line arguments and execute PDB splitting."""
    parser = argparse.ArgumentParser(
        description=f"Enhanced PDB Structure Splitter (v{SCRIPT_VERSION}) - Auto Ligand Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s complex.pdb protein.pdb                    # Auto-generate ligand files
  python %(prog)s complex.pdb protein.pdb --ligand-base myligands
  python %(prog)s input/complex.pdb output/protein.pdb --include-ions
  python %(prog)s --json-config config.json
  python %(prog)s complex.pdb protein.pdb --log-level DEBUG --exclude-cofactors
  python %(prog)s complex.pdb protein.pdb --no-sdf --include-water --include-ions

Key Features:
- NO LONGER REQUIRES ligand output argument - automatically generates ligand filenames
- NEW: --include-ions option to include ions with protein complex
- Handles multiple ligands in a single PDB file
- Creates INDIVIDUAL SDF files for each ligand with detailed properties
- Robust fallback to PDB format when SDF conversion fails
- Detailed ligand inventory and statistics
- Output files placed in input directory when just filenames provided
- Each ligand gets its own SDF file automatically named

Automatic Ligand Naming:
- Default: {input_basename}_ligands_{chain}_{resname}_{resnum}.sdf
- Custom base: {ligand_base_name}_{chain}_{resname}_{resnum}.sdf
"""
    )
    
    parser.add_argument("input_pdb", nargs="?", help="Input PDB file path")
    parser.add_argument("protein_output", nargs="?", help="Output protein PDB file path")
    parser.add_argument("--ligand-base-name", "--ligand-base", type=str, 
                       help="Base name for ligand files (auto-generated if not provided)")
    parser.add_argument("--json-config", type=str, help="JSON configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level (default: INFO)")
    parser.add_argument("--exclude-cofactors", action="store_true", 
                       help="Exclude common cofactors from ligand extraction")
    parser.add_argument("--exclude-buffers", action="store_false", dest="include_buffers",
                       help="Include buffer components as ligands (default: exclude)")
    parser.add_argument("--include-water", action="store_true", 
                       help="Include water molecules in protein output")
    parser.add_argument("--include-ions", action="store_true", 
                       help="Include ions in protein output (creates protein-ion complex)")
    parser.add_argument("--no-sdf", action="store_true", 
                       help="Output ligands in PDB format instead of SDF")
    parser.add_argument("--output-json", type=str, 
                       help="Save results to a specific JSON file")
    parser.add_argument("--version", action="version", 
                       version=f"Enhanced PDB Structure Splitter v{SCRIPT_VERSION}")
    
    args = parser.parse_args()
    
    create_directories()
    
    run_options = collect_run_options(args)
    
    # Initialize configuration variables
    exclude_cofactors = args.exclude_cofactors
    exclude_buffers = not getattr(args, 'include_buffers', False)
    include_water = args.include_water
    include_ions = getattr(args, 'include_ions', False)
    use_sdf = not args.no_sdf
    
    try:
        if args.json_config:
            config = load_config_from_json(args.json_config)
            input_pdb = config.get("input_pdb")
            protein_output = config.get("protein_output")
            ligand_base_name = config.get("ligand_base_name", args.ligand_base_name)
            output_json = config.get("output_json", args.output_json)
            exclude_cofactors = config.get("exclude_cofactors", args.exclude_cofactors)
            exclude_buffers = config.get("exclude_buffers", exclude_buffers)
            include_water = config.get("include_water", args.include_water)
            include_ions = config.get("include_ions", include_ions)
            use_sdf = config.get("use_sdf_format", use_sdf)
            
            run_options.update({
                "config_source": "JSON file",
                "input_file": input_pdb,
                "protein_output": protein_output,
                "ligand_base_name": ligand_base_name,
                "exclude_cofactors": exclude_cofactors,
                "exclude_buffers": exclude_buffers,
                "include_water": include_water,
                "include_ions": include_ions,
                "use_sdf_format": use_sdf
            })
        else:
            if not args.input_pdb or not args.protein_output:
                parser.error("Either provide input_pdb and protein_output arguments, or use --json-config")
            input_pdb = args.input_pdb
            protein_output = args.protein_output
            ligand_base_name = args.ligand_base_name
            output_json = args.output_json
            run_options["config_source"] = "Command line arguments"
        
        if not os.path.exists(input_pdb):
            raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")
        
        # Initialize PDB splitter
        splitter = PDBSplitter(
            log_level=args.log_level,
            exclude_cofactors=exclude_cofactors,
            include_water=include_water,
            include_ions=include_ions,
            exclude_buffers=exclude_buffers
        )
        
        # Process the PDB file
        results = splitter.split_pdb_file(
            input_pdb, protein_output, ligand_base_name, use_sdf=use_sdf
        )
        
        # Save results to JSON
        json_output_path = output_json or os.path.join(DEFAULT_OUTPUT_DIR, "pdb_splitter_results.json")
        save_results_to_json(results, json_output_path, run_options)
        
        # Print final summary
        print("\n" + "="*70)
        print(f"ENHANCED PDB SPLITTING RESULTS (v{SCRIPT_VERSION})")
        print("="*70)
        print(f"Input file: {input_pdb}")
        print(f"Processing successful: {'YES' if results['success'] else 'NO'}")
        print(f"Protein extracted: {'YES' if results['protein_extracted'] else 'NO'}")
        print(f"Ligand extracted: {'YES' if results['ligand_extracted'] else 'NO'}")
        print(f"Ligand format: {results['ligand_format']}")
        print("-" * 35)
        print("Protein Options:")
        print(f"  Include water: {'YES' if include_water else 'NO'}")
        print(f"  Include ions: {'YES' if include_ions else 'NO'}")
        print("-" * 35)
        print("Structure Analysis:")
        stats = results['statistics']
        print(f"  Total residues: {stats['total_residues']}")
        print(f"  Protein residues: {stats['protein_residues']}")
        print(f"  Ligand residues: {stats['ligand_residues']}")
        print(f"  Water molecules: {stats['water_molecules']}")
        print(f"  Ion molecules: {stats['ion_molecules']}")
        print(f"  Cofactor molecules: {stats['cofactor_molecules']}")
        print(f"  Buffer molecules: {stats['buffer_molecules']}")
        print(f"  Protein chains: {stats['protein_chains']}")
        print("-" * 35)
        print("Ligand Details:")
        print(f"  Individual ligands found: {stats['ligand_molecules']}")
        print(f"  Unique ligand types: {stats['unique_ligand_types']}")
        print(f"  Successfully processed: {stats['individual_ligands_processed']}")
        if results['ligand_format'] == 'SDF':
            print(f"  SDF molecules written: {stats['sdf_molecules_written']}")
        
        # Display ligand inventory
        if results.get('ligand_inventory'):
            print("\n  Ligand Inventory:")
            for lig in results['ligand_inventory']:
                print(f"    - {lig['full_id']} ({lig['atom_count']} atoms)")
        
        print("-" * 35)
        print("Output files:")
        if results['protein_extracted']:
            protein_type = "Protein"
            if include_water and include_ions:
                protein_type = "Protein + Water + Ions Complex"
            elif include_water:
                protein_type = "Protein + Water Complex"
            elif include_ions:
                protein_type = "Protein + Ions Complex"
            print(f"  {protein_type}: {protein_output}")
        
        if results['ligand_extracted']:
            if results['ligand_format'] == 'SDF':
                # List individual SDF files created
                sdf_files = results.get('statistics', {}).get('sdf_files_created', [])
                if sdf_files:
                    print(f"  Individual Ligand SDF files ({len(sdf_files)}):")
                    for sdf_file in sdf_files:
                        print(f"    - {sdf_file}")
                else:
                    print(f"  Ligands (SDF): Auto-generated based on input filename")
            else:
                print(f"  Ligands (PDB): Auto-generated based on input filename")
        print(f"  Results JSON: {json_output_path}")
        print("-" * 35)
        print("Dependencies:")
        print(f"  BioPython: {'AVAILABLE' if BIOPYTHON_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  RDKit: {'AVAILABLE' if RDKIT_AVAILABLE else 'NOT AVAILABLE'}")
        print("="*70)
        
        if not results['success']:
            print("\nWARNING: Processing was not fully successful. Check the log for errors.")
            sys.exit(1)
        elif results['ligand_extracted'] and results['ligand_format'] == 'SDF':
            sdf_count = stats.get('sdf_molecules_written', 0)
            print(f"\nSUCCESS: Created {sdf_count} individual SDF files for ligands!")
            if include_ions:
                print("SUCCESS: Protein complex includes ions as requested!")
        
    except Exception as e:
        logging.error(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()