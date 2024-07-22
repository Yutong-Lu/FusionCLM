import os

import numpy as np
import pandas as pd

from typing import List
import wandb

# import molnet loaders from deepchem
# from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21
from rdkit import Chem

# import MolNet dataloder from bert-loves-chemistry fork
# from chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop

import sklearn
import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv("/train_clintox.csv")

valid_df = pd.read_csv("/valid_clintox.csv")

valid2_df = pd.read_csv("/valid2_clintox.csv")

test_df = pd.read_csv("/test_clintox.csv")

# set the logging directories
project_name = "chemberta-2-clintox-stf"
output_path = './clintox-stf-output'
model_name = "77m-MTR"

model_folder = os.path.join(output_path, model_name)

evaluation_folder = os.path.join(output_path, model_name + '_evaluation')
if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)

# set the parameters (default)
EPOCHS = 100
BATCH_SIZE = 64
learning_rate = 1.395993528131019e-05
weight_decay = 0

patience = 10
optimizer = "AdamW"
manual_seed = 4

wandb.login()

# configure Weights & Biases logging
wandb_kwargs = {'name' : 'clintox-1'}

# configure training
classification_args = {'evaluate_each_epoch': True,
                       'evaluate_during_training_verbose': True,
                       'evaluate_during_training' : True,
                       'overwrite_output_dir': True,
                       'best_model_dir' : model_folder,
                       'no_save': False,
                       'save_eval_checkpoints': False,
                       'save_model_every_epoch': False,
                       'save_best_model' : True,
                       'save_steps': -1,
                       'num_train_epochs': EPOCHS,
                       'weight_decay': weight_decay,
                       'use_early_stopping': True,
                       'early_stopping_patience': patience,
                       'early_stopping_delta': 0.001,
                       'early_stopping_metrics': 'eval_loss',
                       'early_stopping_metrics_minimize': True,
                       'early_stopping_consider_epochs' : True,
                       'fp16' : False,
                       'optimizer' : optimizer,
                       'weight_decay': weight_decay,
                       'max_seq_length': 512,
                       'adam_betas' : (0.95, 0.999),
                       'learning_rate' : learning_rate,
                       'manual_seed': manual_seed,
                       'train_batch_size' : BATCH_SIZE,
                       'eval_batch_size' : BATCH_SIZE,
                       'logging_steps' : len(train_df) / BATCH_SIZE,
                       'auto_weights': True, # change to true
                       'wandb_project': project_name,
                       'wandb_kwargs': wandb_kwargs}

model = ClassificationModel('roberta', 'DeepChem/ChemBERTa-77M-MLM', args=classification_args)

results = model.train_model(train_df, eval_df=valid_df, output_dir=model_folder)

# load the best model
model = ClassificationModel('roberta', model_folder, args=classification_args)

# evaluate the best model
result, model_outputs, wrong_predictions = model.eval_model(valid2_df, acc=sklearn.metrics.accuracy_score)

# save the results
valid2_eval_results_path = os.path.join(evaluation_folder, 'valid2_eval_results.txt')
with open(valid2_eval_results_path, 'w+', encoding='latin-1') as file:
    file.write(str(result))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def make_predictions_df(test_df, model_outputs):
    predictions_df = test_df.copy()
    # Original class probabilities/scores
    predictions_df['class_0_probability'] = model_outputs[:, 0]
    predictions_df['class_probability'] = model_outputs[:, 1]
    # Calculate y_pred based on the original scores
    predictions_df['y_pred'] = np.argmax(model_outputs, axis=1)
    # Calculate softmax probabilities
    softmax_probs = softmax(model_outputs)
    predictions_df['softmax_class_0_prob'] = softmax_probs[:, 0]
    predictions_df['softmax_class_prob'] = softmax_probs[:, 1]
    return predictions_df

valid2_predictions_df = make_predictions_df(valid2_df, model_outputs)
valid2_predictions_df_path = os.path.join(evaluation_folder, 'chemberta2_valid2_clintox_predictions.csv')
valid2_predictions_df.to_csv(valid2_predictions_df_path)

print(f"Chemberta-2 clintox valid predictions saved to {valid2_predictions_df_path}")

# evaluate the best model
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

# save the results
test_eval_results_path = os.path.join(evaluation_folder, 'test_eval_results.txt')
with open(test_eval_results_path, 'w+', encoding='latin-1') as file:
    file.write(str(result))

test_predictions_df = make_predictions_df(test_df, model_outputs)
test_predictions_df_path = os.path.join(evaluation_folder, 'chemberta2_test_clintox_predictions.csv')
test_predictions_df.to_csv(test_predictions_df_path)

print(f"Chemberta-2 clintox test predictions saved to {test_predictions_df_path}")

