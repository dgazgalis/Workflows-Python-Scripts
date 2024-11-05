import argparse
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def prepare_pdb(input_file, output_file, ph=7.4):
    # Initialize PDBFixer with the input file
    fixer = PDBFixer(filename=input_file)

    # Add missing residues
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # Add missing atoms
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Remove heterogens (like water molecules)
    fixer.removeHeterogens(True)

    # Add missing hydrogens appropriate for the specified pH
    fixer.addMissingHydrogens(ph)

    # Check for ASN residues that might need flipping
    flipped_residues = check_asn_flips(fixer)

    # Write the prepared structure to a new file
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file, 'w'))

    return flipped_residues

def check_asn_flips(fixer):
    flipped_residues = []
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            if residue.name == 'ASN':
                # This is a simplified check. In practice, you'd need a more
                # sophisticated method to determine if flipping is necessary.
                # Here we're just identifying ASN residues.
                flipped_residues.append(f"{residue.id} {residue.name} {chain.id}")
    return flipped_residues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a PDB file and check for ASN flips")
    parser.add_argument("input_file", help="Input PDB file")
    parser.add_argument("output_file", help="Output PDB file")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for hydrogen addition (default: 7.4)")
    args = parser.parse_args()

    flipped_residues = prepare_pdb(args.input_file, args.output_file, args.ph)
    
    print(f"PDB file prepared and saved as {args.output_file}")
    if flipped_residues:
        print("ASN residues that might need flipping:")
        for residue in flipped_residues:
            print(residue)
    else:
        print("No ASN residues identified for potential flipping.")