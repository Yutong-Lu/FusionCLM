import pandas as pd
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import os

# Names of the different dataset versions
dataset_versions = ['example']

# Path to save the feature CSV files
features_folder_path = './features'

# Initialize the MolBertFeaturizer with the checkpoint path
path_to_checkpoint = './checkpoints/last.ckpt'
f = MolBertFeaturizer(path_to_checkpoint, max_seq_len=512)

# Ensure the output directory exists
if not os.path.exists(features_folder_path):
    os.makedirs(features_folder_path)

# Loop through each dataset version
for name in dataset_versions:
    # List of dataset names
    # datasets = [f'valid2_{name}.csv', f'test_{name}.csv']
    datasets = [f'{name}.csv']

    # Path to the folder containing the datasets
    # folder_path = f'/home/yutonglu/new_datasets/{name}'
    folder_path = "./datasets"

    # Loop through each dataset
    for dataset_name in datasets:
        # Construct the full path to the dataset
        dataset_path = os.path.join(folder_path, dataset_name)
        
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Ensure the 'smiles' column exists
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
        else:
            raise ValueError(f"The dataset {dataset_name} does not have a 'SMILES' column.")

        # Transform the SMILES strings into features
        features, masks = f.transform(smiles_list)

        # Insert the SMILES strings as the first column
        features_df = pd.DataFrame(features)
        features_df.insert(0, 'SMILES', smiles_list)

        # Rename columns
        column_names = ['SMILES'] + [f'molbert_features_{i+1}' for i in range(features_df.shape[1] - 1)]
        features_df.columns = column_names
        
        # Generate a filename for the features CSV, saving in a specified output directory
        features_filename = f"molbert_{dataset_name.split('.')[0]}_features.csv"
        features_file_path = os.path.join(features_folder_path, features_filename)
        
        # Save the features to a CSV file
        features_df.to_csv(features_file_path, index=False)
        
        print(f"Features for {dataset_name} saved to {features_file_path}")
