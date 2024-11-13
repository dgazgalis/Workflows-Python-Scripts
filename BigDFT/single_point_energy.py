from BigDFT import Calculators as C
from BigDFT import Inputfiles as I
from BigDFT import Logfiles as L

def read_xyz_file(filename):
    """Read an XYZ file and return a list of atomic positions."""
    positions = []
    with open(filename, 'r') as file:
        # Skip the first two lines (number of atoms and comment line)
        next(file)
        next(file)
        for line in file:
            parts = line.split()
            if len(parts) == 4:
                symbol = parts[0]
                coords = list(map(float, parts[1:4]))
                positions.append({symbol: coords})
    return positions

# Define the input parameters
input_parameters = {
    'dft': {
        'ixc': 'LDA',
        'itermax': 50,
        'gnrm_cv': 1.0e-4,
        'ncong': 1,
        'idsx': 0
    },
    'perf': {
        'use_gpu_acceleration': True,  # Specify GPU acceleration
        'ncount_cluster_x': 4,
        'accel': 'CUDA'  # Specify the GPU acceleration flavor
    },
    'data': {
        'geocode': 'P',
        'cell': [10.0, 10.0, 10.0],
        'units': 'angstroem'
    }
}

# Read positions from XYZ file
posinp = read_xyz_file('input.xyz')

# Set atomic positions
I.set_atomic_positions(input_parameters, posinp=posinp)

# Create the input file
input_file = I.Inputfile(input_parameters)

# Initialize the calculator
calculator = C.SystemCalculator(omp=4, mpi_run='mpirun -n 4', verbose=True)

# Run the calculation
log = calculator.run(input=input_file, name='xyz_input', run_dir='tmp')

# Print the results
if log:
    print("Energy:", log.energy)
else:
    print("Calculation failed.")