import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt

# Function to read and parse CSV file
def read_csv(file_path):
    df = pd.read_csv(file_path)  # Now assuming first row is header
    
    # Define a function to clean the activity values
    def clean_activity(value):
        if isinstance(value, str):
            if '> 10.0E+03' in value:
                return 'inactive'
            elif 'E+' in value:
                return float(value.replace('E+', 'E+0'))  # Ensure consistent scientific notation
        return value
    
    # Apply the cleaning function to each element of the DataFrame
    df = df.apply(lambda x: x.map(clean_activity))
    
    return df

# Function to create Morgan fingerprints from SMILES
def smiles_to_fingerprint(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except:
        pass
    return None

def smiles_to_comprehensive_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Morgan (ECFP) fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    # Topological fingerprint
    topo_fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048, nBitsPerHash=2)
    
    # MACCS keys
    maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
    
    # Atom Pair fingerprint
    atom_pair_fp = AllChem.GetAtomPairFingerprint(mol)
    
    # Torsion fingerprint
    torsion_fp = AllChem.GetTopologicalTorsionFingerprint(mol)
    
    # Convert all fingerprints to numpy arrays
    morgan_array = np.array(morgan_fp)
    topo_array = np.array(topo_fp)
    maccs_array = np.array(maccs_fp)
    atom_pair_array = np.frombuffer(atom_pair_fp.ToBitString().encode(), 'u1') - ord('0')
    torsion_array = np.frombuffer(torsion_fp.ToBitString().encode(), 'u1') - ord('0')
    
    # Concatenate all fingerprints
    return np.concatenate([morgan_array, topo_array, maccs_array, atom_pair_array, torsion_array])

# Function to calculate similarity matrix
def calculate_similarity_matrix(fingerprints):
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix

# Main function
def main():
    file_path = r'C:\Users\dgazg\Downloads\test.csv'
    
    data = read_csv(file_path)
    
    print("Data columns:")
    print(data.columns)
    
    print("\nSample data:")
    print(data.head())

    # Assuming SMILES are in the 'SMILES' column
    smiles_column = 'SMILES'  # change this if your SMILES column has a different name

    # Remove rows with missing or invalid SMILES
    data = data.dropna(subset=[smiles_column])
    data = data[data[smiles_column].apply(lambda x: isinstance(x, str))]

    print(f"\nNumber of compounds after removing invalid SMILES: {len(data)}")

    print("\nData types in SMILES column:")
    print(data[smiles_column].dtype)

    print("\nFirst few entries of SMILES column:")
    print(data[smiles_column].head(10))

    smiles_list = data[smiles_column].tolist()
    
    # Generate comprehensive fingerprints
    fingerprints = []
    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_comprehensive_fingerprint(smiles)
        if fp is None:
            print(f"Warning: Could not generate fingerprint for entry {i}: {smiles}")
        else:
            fingerprints.append(fp)
    
    print(f"\nSuccessfully generated {len(fingerprints)} comprehensive fingerprints out of {len(smiles_list)} entries")
    
    if len(fingerprints) == 0:
        print("No valid fingerprints. Cannot create similarity matrix.")
        return
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(fingerprints)
    
    print("\nSimilarity Matrix Shape:")
    print(similarity_matrix.shape)
    
    print("\nSample of Similarity Matrix:")
    print(similarity_matrix[:5, :5])

    # Plot the similarity matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Similarity')
    plt.title('Comprehensive Similarity Matrix Heatmap')
    plt.xlabel('Molecule Index')
    plt.ylabel('Molecule Index')
    plt.tight_layout()
    plt.show()
  

if __name__ == '__main__':
    main()
