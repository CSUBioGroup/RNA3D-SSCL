import os
from torch.utils.data import Dataset
from config import max_len
import numpy as np
from ss import Extraxt_ss



def extract_rna_data(pdb_path):
    glycosidic_n_atoms = {'A': 'N9', 'G': 'N9', 'U': 'N1', 'C': 'N1', 'T': 'N1'}
    sequence = ''
    coordinates_dict = {}

    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                key = (residue_number, residue_name)
                if key not in coordinates_dict:
                    coordinates_dict[key] = []

                coordinates_dict[key].append((atom_name, [x, y, z]))

                if atom_name == glycosidic_n_atoms.get(residue_name, ''):
                    sequence += residue_name

    # Process the coordinates and handle missing key atoms
    coordinates = []
    for key, atoms in coordinates_dict.items():
        atom_dict = {name: coord for name, coord in atoms}
        required_atoms = ['P', 'C4\'', glycosidic_n_atoms[key[1]]]
        missing_atoms = [atom for atom in required_atoms if atom not in atom_dict]

        if missing_atoms:
            # Choose alternative atoms if any required atom is missing
            alternative_atoms = [coord for name, coord in atoms if name not in required_atoms]
            if alternative_atoms:
                for missing_atom in missing_atoms:
                    message = f"Missing {missing_atom} for nucleotide {key[1]} at residue {key[0]}. Using alternative atom coordinates."
                    print(message)
                    atom_dict[missing_atom] = alternative_atoms[0]  # Use the first available alternative atom
            else:
                print(f"No alternative atoms available for nucleotide {key[1]} at residue {key[0]}.")
                continue  # Skip this nucleotide if no alternative atoms are available

        # Add the coordinates in the required order
        coordinates.extend([atom_dict[atom] for atom in required_atoms])

    coordinates_array = np.array(coordinates).reshape(-1, 3, 3)  


    # Save the sequence to a FASTA file

    fasta_file_name = os.path.join(os.path.dirname(pdb_path), os.path.splitext(os.path.basename(pdb_path))[0] + '.fasta')
    if not os.path.exists(fasta_file_name):
        with open(fasta_file_name, 'w') as fasta_file:
            fasta_file.write(f'>{os.path.basename(pdb_path)}\n{sequence}\n')

    # Assume ss_feature is a function defined elsewhere that generates secondary structure features
    ss_file_name = os.path.join(os.path.dirname(pdb_path), os.path.splitext(os.path.basename(pdb_path))[0] + 'ss')
    # ss_feature(fasta_file_name, ss_file_name, fasta_file_name)
    savepath = os.path.abspath(ss_file_name) + '.npy'
    if len(sequence)>max_len:
        return sequence,coordinates_array,savepath 
    ss = np.load(os.path.abspath(ss_file_name) + '.npy')
    return sequence, coordinates_array, ss

#solo RNA 
def extract_solorna_data(pdb_path):
    fasta_file = pdb_path.replace(".pdb",".fasta")

    sequence = ""

    with open(fasta_file, "r") as file:
        for line in file:
            if not line.startswith(">"):
                sequence += line.strip()

    coor_path = pdb_path.replace(".pdb","_coor.npy")
    coordinates_array = np.load(coor_path)


    # Assume ss_feature is a function defined elsewhere that generates secondary structure features
    ss_file = pdb_path.replace(".pdb","ss.npy")
    # ss_feature(fasta_file, ss_file, fasta_file)
    Extraxt_ss(fasta_file,ss_file)
    if len(sequence)>max_len:
        return sequence,coordinates_array,ss_file
    ss = np.load(ss_file)
    return sequence, coordinates_array, ss

class RNADataset(Dataset):
    def __init__(self, pdb_folder_path, is_solorna = False):
        self.pdb_files = [os.path.join(pdb_folder_path, f) for f in os.listdir(pdb_folder_path) if f.endswith('.pdb')]
        self.data = []
        for pdb_file in self.pdb_files:
            if is_solorna:
                sequence, coords_array, ss = extract_solorna_data(pdb_file)
            else:
                sequence, coords_array, ss = extract_rna_data(pdb_file)            
            # sequence, coords_array, ss = extract_solorna_data(pdb_file)
            self.data.append((sequence, coords_array, ss))
            # if len(sequence)<=1000:
            #     self.data.append((sequence, coords_array, ss))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, coords_array, ss = self.data[idx]
        return sequence, coords_array, ss