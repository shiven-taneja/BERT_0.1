import datetime

import torch

from pathlib import Path

from dataset import WikitextBertDataset
from model import BERT
from trainer import BertTrainer


BASE_DIR = Path(__file__).resolve().parent

EMB_SIZE = 64
HIDDEN_SIZE = 36
EPOCHS = 4
BATCH_SIZE = 12
NUM_HEADS = 6

CHECKPOINT_DIR = BASE_DIR.joinpath('data/bert_checkpoints')

timestamp = datetime.datetime.utcnow().timestamp()
LOG_DIR = BASE_DIR.joinpath(f'data/logs/bert_experiment_{timestamp}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('CUDA')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Prepare dataset")
    ds = WikitextBertDataset(BASE_DIR.joinpath('data/train_output.csv'), ds_from=0, ds_to=809857)

    bert = BERT(len(ds.vocab), EMB_SIZE, HIDDEN_SIZE, NUM_HEADS).to(device)
    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        print_progress_every=20,
        print_accuracy_every=200,
        batch_size=BATCH_SIZE,
        learning_rate=0.00007,
        epochs=EPOCHS
    )
    trainer.load_checkpoint(CHECKPOINT_DIR.joinpath('bert_epoch2_step-1_1679440342.pt'))
    trainer.print_summary()
    trainer()
