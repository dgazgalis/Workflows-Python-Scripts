#!/usr/bin/env python3
"""
SDF to OpenMM XML Force Field Converter

This script converts SDF files to OpenMM-compatible XML force field files using
the Antechamber/AmberTools workflow. It provides a complete pipeline for:
- Generating AM1-BCC charges using Antechamber
- Creating mol2 and prepi parameter files
- Generating frcmod files with parmchk2
- Running tleap to create prmtop files
- Converting AMBER parameters to OpenMM XML format
- Adding CONECT records to PDB files
- Optional dihedral angle freezing for constrained simulations

The script follows best practices for molecular dynamics force field preparation
and provides comprehensive error handling and logging capabilities.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Script version for tracking and reproducibility
SCRIPT_VERSION = "2.1.0"

# Try to import required libraries
try:
    import parmed
    from parmed import Structure
    PARMED_AVAILABLE = True
except ImportError:
    PARMED_AVAILABLE = False
    print("Warning: ParmEd is not available. Please install it: conda install -c conda-forge parmed")

# Configuration constants
DEFAULT_OUTPUT_DIR = "output"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Supported input file formats
SUPPORTED_FORMATS = {'.sdf', '.mol'}

# AmberTools executables
REQUIRED_EXECUTABLES = ['antechamber', 'parmchk2', 'tleap']


class SDFToOpenMMConverter:
    """
    A class to convert SDF files to OpenMM XML force field files using AmberTools.
    
    This class provides a complete workflow for processing small molecule SDF files
    through the Antechamber/AmberTools pipeline to generate OpenMM-compatible
    force field parameters and structure files.
    """
    
    def __init__(self, log_level: str = "INFO", cleanup_intermediate: bool = True):
        """
        Initialize the SDF to OpenMM converter.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            cleanup_intermediate: Whether to clean up intermediate files after processing
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.cleanup_intermediate = cleanup_intermediate
        
        # Check required dependencies
        self._check_dependencies()
        
        # Statistics tracking
        self.stats = {
            "input_format": "",
            "antechamber_successful": False,
            "parmchk2_successful": False,
            "tleap_successful": False,
            "xml_generation_successful": False,
            "pdb_conect_added": False,
            "dihedrals_frozen": False,
            "intermediate_files_cleaned": False,
            "num_atoms": 0,
            "num_bonds": 0,
            "formal_charge": 0,
            "warnings": [],
            "errors": [],
            "generated_files": []
        }
    
    def setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        # Clear existing handlers to avoid duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=LOG_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
    
    def _check_dependencies(self) -> None:
        """Check for required dependencies and executables."""
        # Check ParmEd
        if not PARMED_AVAILABLE:
            raise ImportError("ParmEd is required but not installed. Please install it: conda install -c conda-forge parmed")
        
        # Check AmberTools executables
        missing_executables = []
        for executable in REQUIRED_EXECUTABLES:
            try:
                result = subprocess.run(['which', executable], capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    missing_executables.append(executable)
            except FileNotFoundError:
                # 'which' command not available (e.g., Windows)
                try:
                    subprocess.run([executable, '--help'], capture_output=True, check=False)
                except FileNotFoundError:
                    missing_executables.append(executable)
        
        if missing_executables:
            raise RuntimeError(f"Required AmberTools executables not found: {', '.join(missing_executables)}. "
                             "Please install AmberTools: conda install -c conda-forge ambertools")
        
        self.logger.info("All required dependencies are available")
    
    def run_antechamber(self, sdf_file: str, resname: str = "LIG", charge: int = 0) -> Tuple[str, str]:
        """
        Run Antechamber to generate AM1-BCC charges, mol2 file, and prepi file.
        
        Args:
            sdf_file: Path to the input SDF file
            resname: Residue name for the ligand (default: "LIG")
            charge: Net charge of the small molecule (default: 0)
        
        Returns:
            Tuple of paths to the output mol2 file and prepi file
            
        Raises:
            subprocess.CalledProcessError: If Antechamber execution fails
            FileNotFoundError: If input SDF file doesn't exist
        """
        if not os.path.exists(sdf_file):
            raise FileNotFoundError(f"Input SDF file not found: {sdf_file}")
        
        output_mol2 = f"{resname}.mol2"
        output_prepi = f"{resname}.prepi"
        
        try:
            self.logger.info(f"Running Antechamber for mol2 generation: {sdf_file} -> {output_mol2}")
            cmd_mol2 = [
                "antechamber", "-i", sdf_file, "-fi", "sdf", 
                "-o", output_mol2, "-fo", "mol2", 
                "-c", "bcc", "-nc", str(charge), "-rn", resname
            ]
            
            result = subprocess.run(cmd_mol2, check=True, capture_output=True, text=True)
            self.logger.debug(f"Antechamber mol2 stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Antechamber mol2 stderr: {result.stderr}")
            
            self.logger.info(f"Running Antechamber for prepi generation: {output_mol2} -> {output_prepi}")
            cmd_prepi = [
                "antechamber", "-i", output_mol2, "-fi", "mol2",
                "-o", output_prepi, "-fo", "prepi",
                "-c", "bcc", "-nc", str(charge), "-rn", resname
            ]
            
            result = subprocess.run(cmd_prepi, check=True, capture_output=True, text=True)
            self.logger.debug(f"Antechamber prepi stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Antechamber prepi stderr: {result.stderr}")
            
            # Verify output files exist
            if not os.path.exists(output_mol2):
                raise FileNotFoundError(f"Antechamber failed to generate {output_mol2}")
            if not os.path.exists(output_prepi):
                raise FileNotFoundError(f"Antechamber failed to generate {output_prepi}")
            
            self.stats["antechamber_successful"] = True
            self.stats["generated_files"].extend([output_mol2, output_prepi])
            self.logger.info(f"Antechamber completed successfully: {output_mol2}, {output_prepi}")
            
            return output_mol2, output_prepi
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Antechamber failed with exit code {e.returncode}: {e.stderr}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error in Antechamber execution: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
    
    def run_parmchk2(self, mol2_file: str) -> str:
        """
        Run parmchk2 to generate the frcmod file.
        
        Args:
            mol2_file: Path to the input mol2 file
        
        Returns:
            Path to the output frcmod file
            
        Raises:
            subprocess.CalledProcessError: If parmchk2 execution fails
            FileNotFoundError: If input mol2 file doesn't exist
        """
        if not os.path.exists(mol2_file):
            raise FileNotFoundError(f"Input mol2 file not found: {mol2_file}")
        
        output_frcmod = os.path.splitext(mol2_file)[0] + ".frcmod"
        
        try:
            self.logger.info(f"Running parmchk2: {mol2_file} -> {output_frcmod}")
            cmd = ["parmchk2", "-i", mol2_file, "-f", "mol2", "-o", output_frcmod]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.debug(f"parmchk2 stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"parmchk2 stderr: {result.stderr}")
            
            # Verify output file exists
            if not os.path.exists(output_frcmod):
                raise FileNotFoundError(f"parmchk2 failed to generate {output_frcmod}")
            
            self.stats["parmchk2_successful"] = True
            self.stats["generated_files"].append(output_frcmod)
            self.logger.info(f"parmchk2 completed successfully: {output_frcmod}")
            
            return output_frcmod
            
        except subprocess.CalledProcessError as e:
            error_msg = f"parmchk2 failed with exit code {e.returncode}: {e.stderr}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error in parmchk2 execution: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
    
    def add_conect_records(self, input_pdb: str, output_pdb: str) -> None:
        """
        Add CONECT records to a PDB file using ParmEd structure analysis.
        
        Args:
            input_pdb: Path to the input PDB file without CONECT records
            output_pdb: Path to the output PDB file with CONECT records
            
        Raises:
            FileNotFoundError: If input PDB file doesn't exist
            RuntimeError: If ParmEd structure loading fails
        """
        if not os.path.exists(input_pdb):
            raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")
        
        try:
            self.logger.info(f"Adding CONECT records: {input_pdb} -> {output_pdb}")
            structure = parmed.load_file(input_pdb)
            
            with open(output_pdb, 'w') as out_file:
                # Copy original content (excluding END records)
                with open(input_pdb, 'r') as in_file:
                    for line in in_file:
                        if not line.startswith('END'):
                            out_file.write(line)
                
                # Add CONECT records
                for atom in structure.atoms:
                    if atom.bonds:  # Only write CONECT if atom has bonds
                        conect_line = f"CONECT{atom.idx+1:5d}"
                        for bond in atom.bonds:
                            other_atom = bond.atom1 if bond.atom2 == atom else bond.atom2
                            conect_line += f"{other_atom.idx+1:5d}"
                        out_file.write(conect_line + "\n")
                
                out_file.write("END\n")
            
            self.stats["pdb_conect_added"] = True
            self.stats["generated_files"].append(output_pdb)
            self.logger.info(f"CONECT records added successfully: {output_pdb}")
            
        except Exception as e:
            error_msg = f"Failed to add CONECT records: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise RuntimeError(error_msg)
    
    def run_tleap(self, mol2_file: str, frcmod_file: str, resname: str = "LIG") -> Tuple[str, str]:
        """
        Run tleap to generate the prmtop file and a compatible PDB file.
        
        Args:
            mol2_file: Path to the input mol2 file
            frcmod_file: Path to the input frcmod file
            resname: Residue name for the ligand (default: "LIG")
        
        Returns:
            Tuple of paths to the output prmtop file and PDB file
            
        Raises:
            subprocess.CalledProcessError: If tleap execution fails
            FileNotFoundError: If input files don't exist
        """
        for file_path in [mol2_file, frcmod_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        prmtop_file = f"{resname}.prmtop"
        inpcrd_file = f"{resname}.inpcrd"
        pdb_no_conect = f"{resname}_no_conect.pdb"
        pdb_with_conect = f"{resname}.pdb"
        tleap_input_file = "tleap.in"
        
        # Create tleap input script
        tleap_input = f"""source leaprc.gaff2
{resname} = loadmol2 {mol2_file}
loadamberparams {frcmod_file}
saveamberparm {resname} {prmtop_file} {inpcrd_file}
savepdb {resname} {pdb_no_conect}
quit
"""
        
        try:
            self.logger.info(f"Running tleap: {mol2_file}, {frcmod_file} -> {prmtop_file}")
            
            # Write tleap input file
            with open(tleap_input_file, "w") as f:
                f.write(tleap_input)
            
            # Run tleap
            cmd = ["tleap", "-f", tleap_input_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.debug(f"tleap stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"tleap stderr: {result.stderr}")
            
            # Verify essential output files exist
            if not os.path.exists(prmtop_file):
                raise FileNotFoundError(f"tleap failed to generate {prmtop_file}")
            if not os.path.exists(pdb_no_conect):
                raise FileNotFoundError(f"tleap failed to generate {pdb_no_conect}")
            
            # Add CONECT records to the PDB file
            self.add_conect_records(pdb_no_conect, pdb_with_conect)
            
            # Clean up intermediate files
            if self.cleanup_intermediate:
                for temp_file in [tleap_input_file, pdb_no_conect]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug(f"Removed temporary file: {temp_file}")
            
            self.stats["tleap_successful"] = True
            self.stats["generated_files"].extend([prmtop_file, inpcrd_file, pdb_with_conect])
            self.logger.info(f"tleap completed successfully: {prmtop_file}, {pdb_with_conect}")
            
            return prmtop_file, pdb_with_conect
            
        except subprocess.CalledProcessError as e:
            error_msg = f"tleap failed with exit code {e.returncode}: {e.stderr}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error in tleap execution: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
        finally:
            # Always try to clean up tleap input file on failure
            if os.path.exists(tleap_input_file):
                try:
                    os.remove(tleap_input_file)
                except:
                    pass
    
    def read_prepi(self, filename: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Read atomic data and bonding information from an AMBER prepi file.

        Args:
            filename: Path to the prepi file

        Returns:
            Tuple containing:
                - atom_data: List of [name, type, charge] for each atom
                - bonds: List of [atom1_name, atom2_name] for each bond
                
        Raises:
            FileNotFoundError: If prepi file doesn't exist
            ValueError: If prepi file format is invalid
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Prepi file not found: {filename}")
        
        try:
            self.logger.info(f"Reading prepi file: {filename}")
            
            with open(filename, "r") as f:
                lines = f.readlines()

            atom_dict = {}  # Maps atom ID numbers to atom names
            atom_data = []  # Stores [name, type, charge] for each atom
            bonds = []  # List of bonds between atoms
            
            for i, line_i in enumerate(lines):
                line_data = line_i.split()
                
                # Process atom lines (contain >10 fields)
                if len(line_data) > 10:
                    atom_id = line_data[0]
                    atom_name = line_data[1]
                    atom_type = line_data[2]
                    bond_id = line_data[4]
                    atom_charge = line_data[10]

                    # Skip dummy atoms
                    if atom_type == "DU":
                        continue

                    atom_dict[atom_id] = atom_name
                    atom_data.append([atom_name, atom_type, atom_charge])
                    
                    # Add bond if not connected to dummy (bond_id > 3 indicates real connection)
                    if int(bond_id) > 3:
                        bond_name = atom_dict[bond_id]
                        bonds.append([atom_name, bond_name])
                
                # Process loop-completion lines
                elif line_i.startswith("LOOP"):
                    for line_j in lines[i + 1:]:
                        if len(line_j.split()) == 2:
                            bonds.append(line_j.split())
                        else:
                            break

            self.stats["num_atoms"] = len(atom_data)
            self.stats["num_bonds"] = len(bonds)
            self.logger.info(f"Read prepi file: {len(atom_data)} atoms, {len(bonds)} bonds")
            
            return atom_data, bonds
            
        except Exception as e:
            error_msg = f"Failed to read prepi file {filename}: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise ValueError(error_msg)
    
    def freeze_dihedral_angles(self, xml_file: str, output_file: Optional[str] = None, 
                                 force_constant: float = 1000.0) -> None:
        """
        Freeze all dihedral angles in the ligand XML file by setting force constants to high values.

        Args:
            xml_file: Path to the input XML file
            output_file: Path to the output XML file. If None, modifies input file in-place
            force_constant: Force constant value in kJ/mol to freeze dihedrals (default: 1000.0)
            
        Raises:
            FileNotFoundError: If input XML file doesn't exist
            ET.ParseError: If XML file is malformed
        """
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        
        try:
            self.logger.info(f"Freezing dihedral angles in XML file: {xml_file}")
            
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find and modify PeriodicTorsionForce elements
            torsion_forces = root.findall(".//PeriodicTorsionForce")
            total_frozen = 0
            
            for torsion_force in torsion_forces:
                for torsion in torsion_force.findall("Proper"):
                    # Set k to high value to freeze the dihedral
                    torsion.set("k", str(force_constant))
                    total_frozen += 1

            # Save the modified XML
            output_path = output_file if output_file else xml_file
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            
            self.stats["dihedrals_frozen"] = True
            self.logger.info(f"Froze {total_frozen} dihedral angles with k={force_constant} kJ/mol")
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error in {xml_file}: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to freeze dihedral angles: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            raise

    def create_ligand_xml(self, prmtop_file: str, prepi_file: str, resname: str = "LIG", 
                          output_file: str = "lig.xml", freeze_dihedrals: bool = False) -> bool:
        """
        Create OpenMM XML force field file from AMBER parameter files.
        
        This function converts AMBER .prmtop and .prepi files to an OpenMM-compatible
        XML force field file that can be used for molecular dynamics simulations.

        Args:
            prmtop_file: Path to the .prmtop file
            prepi_file: Path to the .prepi file  
            resname: Residue name for the small molecule
            output_file: Path to the output XML file
            freeze_dihedrals: Whether to freeze all dihedral angles

        Returns:
            True if successful, False otherwise
        """
        for file_path in [prmtop_file, prepi_file]:
            if not os.path.exists(file_path):
                self.logger.error(f"Input file not found: {file_path}")
                return False
        
        try:
            self.logger.info(f"Creating OpenMM XML file: {output_file}")
            
            # Load AMBER parameter file and convert to OpenMM format
            prmtop = parmed.load_file(prmtop_file)
            openmm_params = parmed.openmm.OpenMMParameterSet.from_structure(prmtop)
            
            # Write temporary XML file
            tmp_xml = os.path.splitext(output_file)[0] + "-tmp.xml"
            openmm_params.write(tmp_xml)

            # Read additional ligand information from prepi file
            atom_data, bond_list = self.read_prepi(prepi_file)

            # Process the temporary XML to create the final version
            self._process_xml_template(tmp_xml, output_file, atom_data, bond_list, resname)
            
            # Clean up temporary file
            if os.path.exists(tmp_xml):
                os.remove(tmp_xml)
            
            # Freeze dihedral angles if requested
            if freeze_dihedrals:
                self.freeze_dihedral_angles(output_file)
            
            # Add metadata to the XML file
            self._add_xml_metadata(output_file, prmtop_file, prepi_file, resname, freeze_dihedrals)
            
            self.stats["xml_generation_successful"] = True
            self.stats["generated_files"].append(output_file)
            self.logger.info(f"OpenMM XML file created successfully: {output_file}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to create OpenMM XML file: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.stats["errors"].append(error_msg)
            return False
    
    def _process_xml_template(self, tmp_xml: str, output_file: str, atom_data: List[List[str]], 
                              bond_list: List[List[str]], resname: str) -> None:
        """
        Process the temporary XML file to create the final OpenMM XML format.
        
        Args:
            tmp_xml: Path to temporary XML file from ParmEd
            output_file: Path to final output XML file
            atom_data: List of atom data from prepi file
            bond_list: List of bonds from prepi file
            resname: Residue name
        """
        with open(tmp_xml, "r") as f:
            tmp_xml_lines = f.readlines()

        with open(output_file, "w") as f:
            # Write header lines
            for line in tmp_xml_lines[:4]:
                f.write(line)

            # Process <AtomTypes> section
            f.write("  <AtomTypes>\n")
            for line in tmp_xml_lines:
                if "<Type " in line:
                    # Parse type data from XML line
                    type_data = {}
                    for x in line.split():
                        if "=" in x:
                            key = x.split("=")[0]
                            item = x.split("=")[1].strip("/>").strip('"')
                            type_data[key] = item

                    # Write new lines for each atom with this type
                    for atom in atom_data:
                        if atom[1] != type_data["class"]:
                            continue
                        new_line = (f'    <Type name="{resname}-{atom[0]}" class="{type_data["class"]}" '
                                    f'element="{type_data["element"]}" mass="{type_data["mass"]}"/>\n')
                        f.write(new_line)
                elif "</AtomTypes>" in line:
                    f.write("  </AtomTypes>\n")
                    break

            # Generate residue template section
            f.write(" <Residues>\n")
            f.write(f'  <Residue name="{resname}">\n')
            
            # Write atoms
            for atom in atom_data:
                f.write(f'   <Atom name="{atom[0]}" type="{resname}-{atom[0]}" charge="{atom[2]}"/>\n')
            
            # Write bonds
            for bond in bond_list:
                f.write(f'   <Bond atomName1="{bond[0]}" atomName2="{bond[1]}"/>\n')
            
            f.write("  </Residue>\n")
            f.write(" </Residues>\n")

            # Write remaining force field sections
            for i, line_i in enumerate(tmp_xml_lines):
                if "<HarmonicBondForce>" in line_i:
                    for line_j in tmp_xml_lines[i:]:
                        # Replace "type" with "class" for OpenMM compatibility
                        f.write(line_j.replace("type", "class"))
                    break
    
    def _add_xml_metadata(self, xml_file: str, prmtop_file: str, prepi_file: str, 
                          resname: str, freeze_dihedrals: bool) -> None:
        """
        Add metadata comments to the XML file.
        
        Args:
            xml_file: Path to XML file
            prmtop_file: Path to source prmtop file
            prepi_file: Path to source prepi file
            resname: Residue name
            freeze_dihedrals: Whether dihedrals were frozen
        """
        try:
            with open(xml_file, 'r') as f:
                content = f.read()
            
            # Create metadata comment
            metadata = f""""""
            
            # Insert metadata after XML declaration
            if content.startswith('<?xml'):
                lines = content.split('\n')
                lines.insert(1, metadata)
                content = '\n'.join(lines)
            else:
                content = metadata + content
            
            with open(xml_file, 'w') as f:
                f.write(content)
            
        except Exception as e:
            self.logger.warning(f"Could not add metadata to XML file: {e}")
            self.stats["warnings"].append(f"XML metadata addition failed: {e}")
    
    def cleanup_intermediate_files(self, keep_files: Optional[List[str]] = None) -> None:
        """
        Clean up intermediate files generated during processing.
        
        Args:
            keep_files: List of files to keep (others will be deleted)
        """
        if not self.cleanup_intermediate:
            return
        
        keep_files = keep_files or []
        
        # Common intermediate file patterns
        intermediate_patterns = [
            "ANTECHAMBER_*",
            "ATOMTYPE.INF",
            "PREP.INF", 
            "leap.log",
            "qout",
            "punch",
            "NEWPDB.PDB",
            "ANTECHAMBER.FRCMOD"
        ]
        
        cleaned_files = []
        
        try:
            import glob
            for pattern in intermediate_patterns:
                for file_path in glob.glob(pattern):
                    if file_path not in keep_files:
                        try:
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                        except OSError as e:
                            self.logger.warning(f"Could not remove {file_path}: {e}")
            
            if cleaned_files:
                self.stats["intermediate_files_cleaned"] = True
                self.logger.info(f"Cleaned up {len(cleaned_files)} intermediate files")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
            self.stats["warnings"].append(f"Cleanup error: {e}")
    
    def process_sdf_to_openmm_xml(self, sdf_file: str, resname: str = "LIG", 
                                  charge: int = 0, freeze_dihedrals: bool = False,
                                  output_xml: Optional[str] = None, 
                                  output_pdb: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete workflow to process an SDF file to OpenMM XML format.
        
        This method runs the complete pipeline:
        1. Antechamber (generate mol2 and prepi files with AM1-BCC charges)
        2. parmchk2 (generate frcmod file for missing parameters)
        3. tleap (generate AMBER topology and coordinate files)
        4. XML conversion (create OpenMM-compatible force field file)
        5. Optional dihedral freezing
        
        Args:
            sdf_file: Path to the input SDF file
            resname: Residue name for the ligand (default: "LIG")
            charge: Net charge of the small molecule (default: 0)
            freeze_dihedrals: Whether to freeze all dihedral angles (default: False)
            output_xml: Path to output XML file (default: {resname}_openmm.xml)
            output_pdb: Path to output PDB file (default: {resname}.pdb)
        
        Returns:
            Dictionary containing processing results and statistics
        """
        self.logger.info(f"Starting SDF to OpenMM XML conversion: {sdf_file}")
        
        # Reset statistics
        self.stats = {key: 0 if isinstance(value, (int, float)) else [] if isinstance(value, list) else False 
                      for key, value in self.stats.items()}
        self.stats["formal_charge"] = charge
        
        # Set default output file names if not provided
        final_output_xml = output_xml or f"{resname}_openmm.xml"
        final_output_pdb = output_pdb or f"{resname}.pdb"
        
        try:
            # Validate input file
            file_path = Path(sdf_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Input SDF file not found: {sdf_file}")
            
            file_extension = file_path.suffix.lower()
            self.stats["input_format"] = file_extension
            
            if file_extension not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {file_extension}. "
                               f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            
            # Step 1: Run Antechamber
            mol2_file, prepi_file = self.run_antechamber(sdf_file, resname, charge)
            
            # Step 2: Run parmchk2
            frcmod_file = self.run_parmchk2(mol2_file)
            
            # Step 3: Run tleap
            prmtop_file, pdb_file_from_tleap = self.run_tleap(mol2_file, frcmod_file, resname)
            
            # Step 4: Create OpenMM XML file
            xml_success = self.create_ligand_xml(prmtop_file, prepi_file, resname, 
                                                final_output_xml, freeze_dihedrals)
            
            # Step 5: Copy/rename output PDB if needed
            if final_output_pdb != pdb_file_from_tleap and os.path.exists(pdb_file_from_tleap):
                import shutil
                shutil.move(pdb_file_from_tleap, final_output_pdb)
                # Update generated files list
                self.stats["generated_files"] = [f if f != pdb_file_from_tleap else final_output_pdb 
                                                 for f in self.stats["generated_files"]]
            
            # Step 6: Cleanup intermediate files
            keep_files = [final_output_xml, final_output_pdb, mol2_file, prepi_file, frcmod_file, prmtop_file]
            self.cleanup_intermediate_files(keep_files)
            
            # Prepare results
            results = {
                "success": xml_success,
                "input_file": sdf_file,
                "output_files": {
                    "xml_template": final_output_xml if xml_success else None,
                    "pdb_file": final_output_pdb if os.path.exists(final_output_pdb) else None,
                    "mol2_file": mol2_file if os.path.exists(mol2_file) else None,
                    "prepi_file": prepi_file if os.path.exists(prepi_file) else None,
                    "frcmod_file": frcmod_file if os.path.exists(frcmod_file) else None,
                    "prmtop_file": prmtop_file if os.path.exists(prmtop_file) else None
                },
                "molecule_properties": {
                    "residue_name": resname,
                    "formal_charge": charge,
                    "num_atoms": self.stats["num_atoms"],
                    "num_bonds": self.stats["num_bonds"],
                    "dihedrals_frozen": freeze_dihedrals and self.stats["dihedrals_frozen"]
                },
                "processing_steps": {
                    "antechamber_successful": self.stats["antechamber_successful"],
                    "parmchk2_successful": self.stats["parmchk2_successful"],
                    "tleap_successful": self.stats["tleap_successful"],
                    "xml_generation_successful": self.stats["xml_generation_successful"],
                    "pdb_conect_added": self.stats["pdb_conect_added"],
                    "intermediate_files_cleaned": self.stats["intermediate_files_cleaned"]
                },
                "statistics": self.stats.copy()
            }
            
            self.logger.info(f"SDF to OpenMM XML conversion completed: {'SUCCESS' if xml_success else 'FAILED'}")
            return results
            
        except Exception as e:
            error_msg = f"SDF to OpenMM XML conversion failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.stats["errors"].append(error_msg)
            
            return {
                "success": False,
                "input_file": sdf_file,
                "error": str(e),
                "statistics": self.stats.copy()
            }


def load_config_from_json(json_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        json_file: Path to JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is malformed
    """
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from: {json_file}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON configuration file not found: {json_file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file {json_file}: {e}", e.doc, e.pos)
    except Exception as e:
        logging.error(f"Error loading JSON configuration: {e}")
        raise


def save_results_to_json(results: Dict[str, Any], output_file: str, 
                         run_options: Dict[str, Any] = None) -> None:
    """
    Save processing results to JSON file with comprehensive metadata.
    
    Args:
        results: Processing results dictionary
        output_file: Path to output JSON file
        run_options: Dictionary of run options and parameters
        
    Raises:
        OSError: If unable to write to output file
    """
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive output structure
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "script_version": SCRIPT_VERSION,
                "parmed_available": PARMED_AVAILABLE,
                "required_executables": REQUIRED_EXECUTABLES
            },
            "run_options": run_options or {},
            "results": results,
            "summary": {
                "conversion_successful": results.get("success", False),
                "input_format": results.get("statistics", {}).get("input_format", ""),
                "xml_template_generated": results.get("output_files", {}).get("xml_template") is not None,
                "pdb_file_generated": results.get("output_files", {}).get("pdb_file") is not None,
                "num_atoms": results.get("molecule_properties", {}).get("num_atoms", 0),
                "num_bonds": results.get("molecule_properties", {}).get("num_bonds", 0),
                "formal_charge": results.get("molecule_properties", {}).get("formal_charge", 0),
                "dihedrals_frozen": results.get("molecule_properties", {}).get("dihedrals_frozen", False),
                "total_errors": len(results.get("statistics", {}).get("errors", [])),
                "total_warnings": len(results.get("statistics", {}).get("warnings", []))
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results to JSON: {e}")
        raise


def collect_run_options(args) -> Dict[str, Any]:
    """
    Collect all run options from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary containing all run options
    """
    return {
        "script_version": SCRIPT_VERSION,
        "input_sdf": getattr(args, 'sdf_file', None),
        "output_xml": getattr(args, 'output', None),
        "output_pdb": getattr(args, 'pdb', None),
        "residue_name": args.resname,
        "formal_charge": args.charge,
        "freeze_dihedrals": args.freeze_dihedrals,
        "json_config_file": getattr(args, 'json_config', None),
        "log_level": getattr(args, 'log_level', 'INFO'),
        "cleanup_intermediate": not getattr(args, 'keep_intermediate', False),
        "command_line": " ".join(sys.argv)
    }


def main():
    """
    Main function to handle command-line arguments and execute SDF to OpenMM conversion.
    """
    parser = argparse.ArgumentParser(
        description=f"SDF to OpenMM XML Force Field Converter (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script converts SDF files to OpenMM XML force field files using AmberTools.
The workflow includes: Antechamber -> parmchk2 -> tleap -> OpenMM XML conversion.

Examples:
  # Basic run, creates a timestamped output directory
  python %(prog)s molecule.sdf -n my_molecule

  # Run with options and keep intermediate files
  python %(prog)s molecule.sdf -r MOL -c -1 --freeze-dihedrals --keep-intermediate

  # Run with options for SACP
  python %(prog)s molecule.sdf --freeze-dihedrals

  # Specify all outputs and prevent creating a run directory
  python %(prog)s molecule.sdf -o custom.xml -p custom.pdb --no-run-directory
  
  # Run from a JSON configuration file
  python %(prog)s --json-config config.json

Supported input formats: {', '.join(SUPPORTED_FORMATS)}
Required executables: {', '.join(REQUIRED_EXECUTABLES)}

The script generates:
 - OpenMM XML force field file
 - PDB file with CONECT records
 - Optional intermediate files (mol2, prepi, frcmod, prmtop)
        """
    )

    parser.add_argument("sdf_file", nargs="?", help="Path to the input SDF file")
    
    # Molecule options
    group = parser.add_argument_group("Molecule options")
    group.add_argument("-r", "--resname", default="LIG", metavar="NAME", 
                       help="Residue name for the ligand (default: %(default)s)")
    group.add_argument("-c", "--charge", type=int, default=0, metavar="INT", 
                       help="Net charge of the small molecule (default: %(default)s)")
    group.add_argument("-fd", "--freeze-dihedrals", action="store_true", 
                       help="Freeze all dihedral angles in the ligand")

    # Output options
    group = parser.add_argument_group("Output options")
    group.add_argument("-o", "--output", metavar="FILE", 
                       help="Output XML file name (default: <resname>_openmm.xml)")
    group.add_argument("-p", "--pdb", metavar="FILE", 
                       help="Output PDB file name (default: <resname>.pdb)")
    group.add_argument("--output-json", metavar="FILE",
                       help="Save results to JSON file (default: <run_dir>/conversion_results.json)")
    group.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                       help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})")
    group.add_argument("-n", "--name", dest="run_name", type=str, 
                       help="Run name prefix for directory")
    group.add_argument("--no-run-directory", action="store_true", 
                       help="Don't create a timestamped run directory for outputs")

    # Processing options
    group = parser.add_argument_group("Processing options")
    group.add_argument("--json-config", metavar="FILE", 
                       help="JSON configuration file path")
    group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level (default: %(default)s)")
    group.add_argument("--keep-intermediate", action="store_true", 
                       help="Keep intermediate files from AmberTools")

    # Information options
    group = parser.add_argument_group("Information")
    group.add_argument("--version", action="version", 
                       version=f"%(prog)s v{SCRIPT_VERSION}")

    args = parser.parse_args()
    
    # Initialize logger with default level
    logging.basicConfig(level=args.log_level.upper(), format=LOG_FORMAT)
    
    # Load config from JSON if provided, CLI args will override
    config = {}
    if args.json_config:
        try:
            config = load_config_from_json(args.json_config)
        except Exception as e:
            logging.error(f"Failed to load JSON config: {e}")
            sys.exit(1)

    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_args)

    # Finalize parameters from config
    sdf_file = config.get("sdf_file")
    resname = config.get("resname", "LIG")
    charge = config.get("charge", 0)
    freeze_dihedrals = config.get("freeze_dihedrals", False)
    cleanup_intermediate = not config.get("keep_intermediate", False)
    log_level = config.get("log_level", "INFO")
    output_xml_arg = config.get("output")
    output_pdb_arg = config.get("pdb")
    output_json_arg = config.get("output_json")

    if not sdf_file:
        parser.error("An input sdf_file must be provided via command line or JSON config.")
        
    run_options = collect_run_options(args)
    original_cwd = Path.cwd()
    run_dir = None # Will store the absolute path to the run directory

    try:
        # --- Directory and Path Setup ---
        sdf_file = str(Path(sdf_file).resolve())

        if not config.get('no_run_directory', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = config.get("run_name")
            workflow_name = config.get("workflow_name", "sdf_converter")
            dir_name = f"{run_name}_openmm_{workflow_name}_{timestamp}" if run_name else f"openmm_{workflow_name}_{timestamp}"

            base_output_dir = config.get('output_dir', DEFAULT_OUTPUT_DIR)
            
            # Create and resolve the run directory path BEFORE changing directory
            run_dir = (Path(base_output_dir) / dir_name).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created run directory: {run_dir}")

            # Define output file basenames to be used within the run_dir
            output_xml = Path(output_xml_arg).name if output_xml_arg else f"{resname}_openmm.xml"
            output_pdb = Path(output_pdb_arg).name if output_pdb_arg else f"{resname}.pdb"
            output_json_name = Path(output_json_arg).name if output_json_arg else "conversion_results.json"
            json_output_path = run_dir / output_json_name
        
        else: # No run directory, use paths as given or generate in default output dir
            base_dir = Path(config.get('output_dir', '.')).resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            
            output_xml = base_dir / (output_xml_arg or f"{resname}_openmm.xml")
            output_pdb = base_dir / (output_pdb_arg or f"{resname}.pdb")
            json_output_path = base_dir / (output_json_arg or "conversion_results.json")

        # Initialize converter
        converter = SDFToOpenMMConverter(
            log_level=log_level,
            cleanup_intermediate=cleanup_intermediate
        )
        
        # Change to run directory if it was created
        if run_dir:
            os.chdir(run_dir)

        # Process the SDF file
        results = converter.process_sdf_to_openmm_xml(
            sdf_file=sdf_file,
            resname=resname,
            charge=charge,
            freeze_dihedrals=freeze_dihedrals,
            output_xml=str(output_xml), # Pass basename or full path
            output_pdb=str(output_pdb)  # Pass basename or full path
        )
        
        # If a run directory was used, resolve the relative paths in the results
        if run_dir:
            results['run_directory'] = str(run_dir)
            if "output_files" in results and results["output_files"]:
                for key, path in results["output_files"].items():
                    if path:
                        # path is a basename, join it with the absolute run_dir
                        results["output_files"][key] = str(run_dir / path)
            if "statistics" in results and "generated_files" in results["statistics"]:
                results["statistics"]["generated_files"] = [
                    str(run_dir / f) for f in results["statistics"]["generated_files"]
                ]
        
        # Save results to JSON
        save_results_to_json(results, str(json_output_path), run_options)

    except Exception as e:
        logging.getLogger(__name__).error(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Always change back to original directory
        os.chdir(original_cwd)

    # --- Print Final Summary ---
    print("\n" + "="*70)
    print(f"SDF TO OPENMM XML CONVERSION RESULTS (v{SCRIPT_VERSION})")
    print("="*70)
    print(f"Input file: {results.get('input_file')}")
    if results.get('run_directory'):
        print(f"Run Directory: {results['run_directory']}")
    print(f"Conversion successful: {'YES' if results.get('success') else 'NO'}")
    print("-" * 40)
    print("Processing Steps:")
    steps = results.get('processing_steps', {})
    print(f"  Antechamber: {'SUCCESS' if steps.get('antechamber_successful', False) else 'FAILED'}")
    print(f"  parmchk2: {'SUCCESS' if steps.get('parmchk2_successful', False) else 'FAILED'}")
    print(f"  tleap: {'SUCCESS' if steps.get('tleap_successful', False) else 'FAILED'}")
    print(f"  XML generation: {'SUCCESS' if steps.get('xml_generation_successful', False) else 'FAILED'}")
    print(f"  CONECT records: {'ADDED' if steps.get('pdb_conect_added', False) else 'NOT ADDED'}")
    print(f"  Cleanup: {'DONE' if steps.get('intermediate_files_cleaned', False) else 'SKIPPED'}")
    print("-" * 40)
    print("Molecule Properties:")
    mol_props = results.get('molecule_properties', {})
    print(f"  Residue name: {mol_props.get('residue_name', 'N/A')}")
    print(f"  Formal charge: {mol_props.get('formal_charge', 0)}")
    print(f"  Number of atoms: {mol_props.get('num_atoms', 0)}")
    print(f"  Number of bonds: {mol_props.get('num_bonds', 0)}")
    print(f"  Dihedrals frozen: {'YES' if mol_props.get('dihedrals_frozen', False) else 'NO'}")
    print("-" * 40)
    print("Output Files:")
    output_files = results.get('output_files', {})
    if output_files:
        for file_type, file_path in output_files.items():
            if file_path:
                print(f"  {file_type.replace('_', ' ').title()}: {file_path}")
    print(f"  Results JSON: {json_output_path}")

    stats = results.get('statistics', {})
    if stats.get('warnings'):
        print("-" * 40)
        print("Warnings:")
        for warning in stats['warnings']:
            print(f"  • {warning}")
    
    if stats.get('errors'):
        print("-" * 40)
        print("Errors:")
        for error in stats['errors']:
            print(f"  • {error}")
    
    print("="*70)
    
    if not results.get('success'):
        print("\nWARNING: Conversion was not fully successful. Check the log for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()