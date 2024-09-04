# Workflow Python Scripts
General purpose molceular modling scripts

# Enviroments (conda_envs_yaml)
This repository includes several Python environments in which the scripts were developed and tested. Each environment is defined by a YAML file, which can be used to recreate the environment on your system. There are two types of YAML files for each environment:

General Installation YAML: This file is designed for a general install into a new system, ensuring that all necessary dependencies are included.
Development YAML: This file is tailored to recreate the exact Python environment used for developing new scripts, including specific versions of libraries and tools.

I recomend Mamba for handling enviroments due to the complexity of the enviroments. 

# Enviroment Details
Below is a list of the main libraries and Python versions for each environment:

1) openmm-openff-openfe-grand-py311

    The openmm-openff-openfe-grand-py311 environment is streamlined for computational chemistry and molecular modeling, focusing on Python 3.11. It primarily includes the following core dependencies:

        Python Version: 3.11
        Channels: Includes repositories from omnia, anaconda, conda-forge, essexlab, and defaults.

        OpenMM: A powerful toolkit for molecular simulations.
        OpenFF Toolkit: Essential for creating and manipulating molecular mechanics force fields.
        Grand: Tools for conducting grand canonical Monte Carlo simulations.
        PySCF: An ab initio computational chemistry package for quantum chemical calculations.
     
    These components form the backbone of the environment, suitable for researchers and developers engaged in molecular simulations, force field manipulation, and quantum chemistry computations.

# Tools
    The `tools` folder contains utility scripts designed to assist with various checks and verifications related to the Python environment and system capabilities. These scripts are essential for ensuring that the computational environment meets the requirements for running the molecular modeling scripts. They are also a useful debuging tool. 

# Usage
Most files in this repository are designed to be easily used from the command line, providing a straightforward interface for executing various molecular modeling tasks. Each script is tailored to perform specific functions, such as setting up simulations, analyzing results, or managing environments. Non-CLI variants are also available. These alternatives are meant to allow for easier intergation into existing scripts. By offering these non-CLI alternatives, we aim to support a seamless integration of our scripts into diverse scientific workflows, enhancing both flexibility and productivity.

    ## Command Line Interface (CLI)
    - **Example Command**: `python script_name.py`
    - **Description**: This command runs the specified script and will reult in the help text being printed. Detailed usage instructions can be found in the individual script documentation.

    ## Non-CLI Variants
    - **API Usage and Integration**: Non-CLI variants are designed to facilitate easier integration into existing Python scripts and workflows. These include APIs and modular functions that can be imported and utilized within your code. Examples of how to integrate these functions are provided in the `examples` directory, demonstrating how to incorporate the scripts' functionality into larger projects without needing to use the command line.
    - **Interactive Notebooks for Exploration**: Jupyter notebooks in the `notebooks` directory serve as interactive tools for exploring the scripts' capabilities. While they include hard-coded inputs and outputs for educational purposes, they also illustrate how the scripts can be adapted and integrated into more complex workflows. These notebooks are particularly useful for understanding the underlying logic and for prototyping new integrations.

