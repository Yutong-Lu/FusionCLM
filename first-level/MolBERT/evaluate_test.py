from molbert.models.finetune import FinetuneSmilesMolbertModel
import pytorch_lightning as pl

mod = './checkpoints/last.ckpt'
data = 'example.csv'

models = [mod]
test_sets = [data]

for mod, test in zip(models, test_sets):
    # Load the model
    model = FinetuneSmilesMolbertModel.load_from_checkpoint(mod)
    trainer = pl.Trainer()
    trainer.test(model)
    print(f"I have run the test script for test with model {mod} and test set {test}.")
