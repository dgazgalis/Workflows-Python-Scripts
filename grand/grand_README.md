# Grand: Grand Canonical Sampling of Waters in OpenMM

## Description
The `grand` folder contains scripts and tools designed for Simulated Annealing of the Chemical Potential (SACP), a computational technique used in molecular modeling to control the chemical potential of particles in a system. SACP is particularly useful in grand canonical Monte Carlo simulations, where the number of particles can fluctuate, allowing for the study of systems under different conditions of particle exchange.

## What is Simulated Annealing of the Chemical Potential (SACP)?
Simulated Annealing of the Chemical Potential (SACP) is a computational method used to adjust the chemical potential of a system to achieve a desired equilibrium state. It is particularly useful for determining free energies of binding for small molecular fragments and controlling particle density in simulations.

## When to Run SACP
SACP is typically run when:
- Equilibrium and Density Control: When you need to maintain specific particle densities in grand canonical ensemble simulations, study system behavior under different chemical potentials, or determine equilibrium states for systems with fluctuating particle numbers.
- Binding Studies and Free Energy Calculations: For calculating binding free energies of small molecular fragments, exploring multiple binding modes, and relating simulations to experimental concentrations. This is crucial for site identification, de novo design, and lead optimization in drug discovery.
- Complex Energy Landscapes: When dealing with systems that have challenging energy landscapes, where overcoming energy barriers and avoiding local minima traps is essential for accurate sampling.
- Solvation and Environmental Effects: For investigating solvation effects and maintaining correct solvent density while studying solute behaviors in various environments.

## Usage
To use the SACP scripts in this folder, ensure you have the necessary dependencies installed. Refer to the `conda_envs_yaml/openmm-openff-openfe-grand-py311.yaml` file for the required Python environment and packages. 
Refer to each README for more infomation on system preperation and running the simulations. 

## Refrences
GRAND
1. G. A. Ross, M. S. Bodnarchuk, J. W. Essex, J. Am. Chem. Soc. 2015, 137, 47, 14930-14943, DOI: https://doi.org/10.1021/jacs.5b07940
2. G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo Martinez, J. W. Essex, J. Chem. Theory Comput. 2017, 13, 12, 6373-6381, DOI: https://doi.org/10.1021/acs.jctc.7b00738
3. M. L. Samways, H. E. Bruce Macdonald, J. W. Essex, _J. Chem. Inf. Model._, 2020, 60, 4436-4441, DOI: https://doi.org/10.1021/acs.jcim.0c00648

SACP
1. F. Guarnieri, M. Mezei, J. Am. Chem. Soc. 1996, 118, 35, 8493-8494, DOI: https://doi.org/10.1021/ja961482a
2. M. Clark, F. Guarnieri, I. Shkurko, J. Wiseman, J. Chem. Inf. Model. 2006, 46, 1, 231-242, DOI: https://doi.org/10.1021/ci050268f
