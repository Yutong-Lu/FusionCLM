import torch
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import pandas as pd

from transformers import pipeline

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np

import wandb

# from tdc import Evaluator

# Load the dataset
train_df = pd.read_csv("train_delaney.csv")
valid2_df = pd.read_csv("valid2_delaney.csv")
valid_df = pd.read_csv("valid_delaney.csv")
test_df = pd.read_csv("test_delaney.csv")

print(f"There are {len(train_df)} molecules in Train df.")
print(f"There are {len(valid2_df)} molecules in Valid2 df.")
print(f"There are {len(valid_df)} molecules in Val df.")
print(f"There are {len(test_df)} molecules in Test df.")

device = "cuda"
label = "logSolubility"

class Input(Dataset):
    def __init__(self, data, tokenizer, max_length, labels_mean=None, labels_std=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Use provided mean and std for normalization
        self.labels_mean = labels_mean
        self.labels_std = labels_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]["smiles"]
        inputs = self.tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs["input_ids"] = inputs["input_ids"].squeeze(0).to(device)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0).to(device)
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0).to(device)

        # Check if mean and std are provided before normalization
        label = self.data.iloc[idx]["logSolubility"]
        if self.labels_mean is not None and self.labels_std is not None:
            normalized_label = (label - self.labels_mean) / self.labels_std
        else:
            normalized_label = label
        
        inputs["labels"] = torch.tensor(normalized_label, dtype=torch.float).unsqueeze(0).to(device)

        return inputs


# Load a pretrained transformer model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# config.num_hidden_layers += 3
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

max_length = 512
# Prepare the dataset for training

training_mean = train_df[label].mean()
training_std = train_df[label].std()

# training_mean = 0
# training_std = 1

train_dataset = Input(train_df, tokenizer, max_length, labels_mean=training_mean, labels_std=training_std)
print(f"this is the first training set label: {train_dataset[0]['labels']} with mean {train_dataset.labels_mean} and sd {train_dataset.labels_std}.")

val_dataset = Input(valid_df, tokenizer, max_length)
print(f"this is the first val set label: {val_dataset[0]['labels']} and the label of val set is not normalized with mean not calculated {val_dataset.labels_mean}.")

class RegressionTrainer(Trainer):
    def __init__(self, labels_mean=None, labels_std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_mean = labels_mean
        self.labels_std = labels_std

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs[0]  
        # the following two are the same
        # print(outputs.logits)
        # print(predictions)

        if model.training:
            # During training, compare predictions directly with normalized labels
            loss = torch.sqrt(mse_loss(predictions, labels))
        else:
            # During evaluation, reverse normalize predictions before calculating RMSE with original labels
            reverse_normalized_predictions = predictions * self.labels_std + self.labels_mean
            loss = torch.sqrt(mse_loss(reverse_normalized_predictions, labels))
        
        return (loss, outputs) if return_outputs else loss

project = "Chemberta2-delaney-finetune"
display_name = "delaney-1"

wandb.init(project=project, name=display_name)

EPOCH = 100
learning_rate = 5e-05
batch_size = 8
weight_decay = 0

# Set up training arguments
training_args = TrainingArguments(
    dataloader_pin_memory=False,
    overwrite_output_dir=True,
    output_dir="./delaney-tf-output",
    num_train_epochs=EPOCH,      
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_steps=15,
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    logging_dir="./delaney-tf-output/logs",
    load_best_model_at_end=True,
    do_train=True,
    do_eval=True,
    seed=4,                      # Set a random seed for reproducibility
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]  # Adjust according to your model output format
    labels = labels.flatten()

    # During evaluation, reverse normalize predictions before calculating metrics
    eval_predictions = predictions * training_std + training_mean
    eval_r2 = r2_score(labels, eval_predictions)
    eval_mse = mean_squared_error(labels, eval_predictions)
    eval_mae = mean_absolute_error(labels, eval_predictions)
    eval_pearson_coef = pearsonr(labels, eval_predictions)[0]
    eval_rmse = torch.sqrt(torch.tensor(eval_mse))
    return {
        'eval_r2': eval_r2,
        'eval_rmse': eval_rmse.item(),  # Converting to Python scalar
        'eval_pearson': eval_pearson_coef,
        'eval_mae': eval_mae
    }

# Train the model
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    labels_mean=training_mean,
    labels_std=training_std,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./mod-tf-output") # save model to output folder

# Create a prediction pipeline
predictor = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Evaluate on the training set for meta model
valid2_smiles = valid2_df['smiles']

model.eval()

# Predict properties for new SMILES strings
valid2_z_preds = []
valid2_raw_preds = []
for smiles in valid2_smiles:
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the raw prediction from the logits
    raw_prediction = outputs.logits.squeeze().item()
    
    # Apply reverse normalization to the raw prediction
    reverse_normalized_prediction = (raw_prediction * training_std) + training_mean
    
    # Store predictions in separate lists
    valid2_z_preds.append(raw_prediction)
    valid2_raw_preds.append(reverse_normalized_prediction)

# Calculate metrics for reverse-normalized predictions as these are on the correct scale
valid2_r2 = r2_score(valid2_df[label], valid2_raw_preds)
valid2_rmse = np.sqrt(mean_squared_error(valid2_df[label], valid2_raw_preds))
valid2_corr, _ = pearsonr(valid2_df[label], valid2_raw_preds)

print("Root Mean Squared Error:", valid2_rmse)
print("R2:", valid2_r2)
print("Correlation:", valid2_corr)

# Create a DataFrame including both raw and reverse-normalized predictions
df = pd.DataFrame({
    'target': valid2_df[label],
    'target_z': (valid2_df[label] * training_std) + training_mean,
    'pred_z': valid2_z_preds,
    'pred_raw': valid2_raw_preds
})

# Saving the DataFrame with the correct column names to a new CSV file
csv_file_path = './chemberta2_valid2_delaney_predictions.csv'
df.to_csv(csv_file_path, index=False)
print(f"Successfully saved to {csv_file_path}")

# Evaluate on test set

# Prepare new SMILES strings for prediction
test_smiles = test_df['smiles']

model.eval()

# Predict properties for new SMILES strings
test_z_predictions = []
test_raw_predictions = []
for smiles in test_smiles:
    inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    raw_prediction = outputs.logits.squeeze().item()
    reverse_normalized_prediction = (raw_prediction * training_std) + training_mean
    
    test_z_predictions.append(raw_prediction)
    test_raw_predictions.append(reverse_normalized_prediction)

# Since we're comparing normalized values, ensure your predictions are on the correct scale as expected
test_r2 = r2_score(test_df[label], test_raw_predictions)
test_rmse = np.sqrt(mean_squared_error(test_df[label], test_raw_predictions))
test_corr, _ = pearsonr(test_df[label], test_raw_predictions)

print("Root Mean Squared Error:", test_rmse)
print("R2:", test_r2)
print("Correlation:", test_corr)

# Create a DataFrame including both raw and reverse-normalized predictions
df = pd.DataFrame({
    'target': test_df[label],
    'target_z': (test_df[label] * training_std) + training_mean,
    'pred_z': test_z_predictions,
    'pred_raw': test_raw_predictions
})

# Saving the DataFrame with the correct column names to a new CSV file
csv_file_path = './chemberta2_test_delaney_predictions.csv'
df.to_csv(csv_file_path, index=False)
print(f"Successfully saved to {csv_file_path}")