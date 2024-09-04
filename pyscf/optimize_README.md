# optimize.py

## Description
`optimize.py` is a Python script designed for optimizing molecular geometries using Density Functional Theory (DFT). This script leverages the PySCF library, with optional GPU acceleration through `gpu4pyscf`, to perform efficient and accurate geometry optimizations. It supports various functionalities such as specifying molecular charge, spin multiplicity, basis set, DFT functional, and the number of CPU cores to use. Additionally, it provides options for optimizing only hydrogen atoms and includes robust error handling and logging mechanisms.

# Features
- **DFT Geometry Optimization**: Utilizes DFT for optimizing molecular geometries.
- **GPU Acceleration with Automatic CPU Fallback**: Optionally uses GPU acceleration for faster calculations. If GPU acceleration fails, the script automatically falls back to CPU computation.
- **Error Handling**: Includes robust error handling and logging for debugging and reliability.
- **Customizable Parameters**: Allows specification of molecular charge, spin multiplicity, basis set, DFT functional, and number of CPU cores.
- **Hydrogen Optimization**: Option to optimize only hydrogen atoms.
- **Output Formats**: Saves optimized geometries in both XYZ and SDF formats.

# Requirements
- Python 3.11
- PySCF
- gpu4pyscf (optional for GPU acceleration)
- RDKit
- NumPy
- CuPy (optional for GPU acceleration)

# Enviroment
Run in the openmm-openff-openfe-grand-py311 enviroment. 