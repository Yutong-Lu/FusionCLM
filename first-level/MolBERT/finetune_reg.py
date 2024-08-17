import pandas as pd
from molbert.apps.finetune import FinetuneSmilesMolbertApp

def finetune(
    train_path,
    valid_path,
    test_path,
    mode,
    learning_rate,
    weight_decay,
    batch_size,
    fast_dev_run,
    label_column,
    train_mean,
    train_std,
    pretrained_model_path,
    max_epochs,
    output_dir,
):
    """
    This function runs finetuning for given arguments.

    Args:
        dataset: Name of the MoleculeNet dataset, e.g. BBBP
        train_path: file to the csv file containing the training data
        valid_path: file to the csv file containing the validation data
        test_path: file to the csv file containing the test data
        mode: either regression or classification
        label_column: name of the column in the csv files containing the labels
        pretrained_model_path: path to a pretrained molbert model
        max_epochs: how many epochs to run at most
        freeze_level: determines what parts of the model will be frozen. More details are given in molbert/apps/finetune.py
        learning_rate: what learning rate to use
        num_workers: how many workers to use
        batch_size: what batch size to use for training
    """

    # default_path = os.path.join('./logs/', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
    # output_dir = os.path.join(default_path, dataset)
    raw_args_str = (
        f"--max_seq_length 512 "
        f"--max_epochs {max_epochs} "
        f"--fast_dev_run {fast_dev_run} "
        f"--train_file {train_path} "
        f"--valid_file {valid_path} "
        f"--test_file {test_path} "
        f"--mode {mode} "
        f"--output_size {1 if mode == 'regression' else 2} "
        f"--pretrained_model_path {pretrained_model_path} "
        f"--label_column {label_column} "
        f"--learning_rate {learning_rate} "
        f"--weight_decay {weight_decay} "
        f"--batch_size {batch_size} "
        f"--train_mean {train_mean} "
        f"--train_std {train_std} "
        f"--gpus 1 "
        f"--default_root_dir {output_dir}"
    )

    raw_args = raw_args_str.split(" ")

    lightning_trainer = FinetuneSmilesMolbertApp().run(raw_args)
    return lightning_trainer

name = 'delaney'
label_column = 'logSolubility'
train_data_path = f'./molbert/train_{name}.csv'

# Load the training data
train_data = pd.read_csv(train_data_path)

# Compute statistics for the label
train_mean = train_data[label_column].mean()
train_std = train_data[label_column].std()

# Report train mean and std
print(f"Training Mean: {train_mean}")
print(f"Training Standard Deviation: {train_std}")

# Standardize the label
train_data[label_column] = (train_data[label_column] - train_mean) / train_std

# Save the standardized data
standardized_train_path = train_data_path.replace('train_', 'standardized_train_')
train_data.to_csv(standardized_train_path, index=False)

# Use the standardized data in the tune function
finetune(
    train_path=standardized_train_path,
    valid_path=f'./molbert/valid_{name}.csv',
    test_path=f'./molbert/valid2_{name}.csv',
    label_column=label_column,
    mode='regression',
    learning_rate=3e-05,
    weight_decay=0.01,
    batch_size=32,
    max_epochs=30,
    train_mean=train_mean,
    train_std=train_std,
    fast_dev_run=0,
    pretrained_model_path='./MolBERT/pretrained/molbert_100epochs/checkpoints/last.ckpt',
    output_dir=f'./MolBERT/finetune/{name}',
)

print(f"I have run the script for reg finetuning.")