import time
from datetime import datetime
from pathlib import Path

import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_childes import WikitextBertDataset
from model import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def percentage(batch_size: int, max_index: int, current_index: int):
    """Calculate epoch progress percentage
    Args:
        batch_size: batch size
        max_index: max index in epoch
        current_index: current index
    Returns:
        Passed percentage of dataset
    """
    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    """Calculate NSP accuracy between two tensors
    Args:
        result: result calculated by model
        target: real target
    Returns:
        NSP accuracy
    """
    s = (result.argmax(1) == target.argmax(1)).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor):
    """Calculate MLM accuracy between ONLY masked words
    Args:
        result: result calculated by model
        target: real target
        inverse_token_mask: well-known inverse token mask
    Returns:
        MLM accuracy
    """
    r = result.argmax(-1).masked_select(~inverse_token_mask)
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)


class BertTrainer:
    """
    Training, checkpointing, and loading the BERT model

    Args: 
        model (nn.Module): The BERT model
        dataset (torch.utils.data.Dataset): The dataset used 
        log_dir (Path): Directory where the log files will be saved 
        checkpoint_dir (Path): Directory where the checpoints will be saved 
        print_progress_every (int): After how many batches should the training progress be printed
        print_accuracy_every (int): After how many batches should the accuracy be printed
        batch_size (int): Size of the batches
        learining_rate (float): Step size of each iteration
        epochs (int): Number of epochs to run 

    Attributes:
         model (nn.Module): Where we store the BERT model 
         dataset (torch.utils.data.Dataset): Where we store the dataset
         batch_size (int): Where we store the batch size 
         epochs (int): Where we store the epochs 
         current_epochs (int): A counter for the current epoch
         loader (DataLoader): Loading the data 
         writer (SummaryWriter): Logging data for TensorBoard 
         checkpoint_dir (Path): Where we store the checkpoint directory 
         criterion (nn.BCEWithLogitsLoss): Loss function
         ml_criterion (nn.NLLLoss): Loss function
         optimizer (torch.optim.Adam): Adam optimizer 
    #TODO change dataset
    """

    def __init__(self,
                 model: BERT,
                 dataset: WikitextBertDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5,
                 ):
        self.model = model
        self.dataset = dataset

        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.ml_criterion = nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)

        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def print_summary(self):
        """Print a summary of the training configuration."""
        ds_len = len(self.dataset)

        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Max / Optimal sentence len: {self.dataset.optimal_sentence_length}")
        print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()

    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)

    def train(self, epoch: int):
        """
        Train the BERT model for one epoch
        Args:
            epoch (int): The current epoch

        Return:
            loss (float): Combined loss from NSP and MLM
        """
        print(f"Begin epoch {epoch}")

        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, value in enumerate(self.loader):
            index = i + 1
            inp, mask, inverse_token_mask, token_target, nsp_target = value
            self.optimizer.zero_grad()

            token, nsp = self.model(inp, mask)

            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill(tm, 0)

            loss_token = self.ml_criterion(token.transpose(1, 2), token_target)  # 1D tensor as target is required
            loss_nsp = self.criterion(nsp, nsp_target)

            loss = loss_token + loss_nsp
            average_nsp_loss += loss_nsp
            average_mlm_loss += loss_token

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                s = self.training_summary(elapsed, index, average_nsp_loss, average_mlm_loss)

                if index % self._accuracy_every == 0:
                    s += self.accuracy_summary(index, token, nsp, token_target, nsp_target, inverse_token_mask)

                print(s)

                average_nsp_loss = 0
                average_mlm_loss = 0
        return loss

    def training_summary(self, elapsed, index, average_nsp_loss, average_mlm_loss):
        """
        Create a summary of the training progress

        Args: 
            elapsed (time.struct_time): Elapsed time 
            index (int): Current batch index 
            average_nsp_loss (float): Accumulated NSP Loss 
            average_mlm_loss (float): Accumulated MLM Loss
        
        Return: 
            s (string): String of the summary of the training progress
        """
        passed = percentage(self.batch_size, self._ds_len, index)
        global_step = self.current_epoch * len(self.loader) + index

        print_nsp_loss = average_nsp_loss / self._print_every
        print_mlm_loss = average_mlm_loss / self._print_every

        s = f"{time.strftime('%H:%M:%S', elapsed)}"
        s += f" | Epoch {self.current_epoch + 1} | {index} / {self._batched_len} ({passed}%) | " \
             f"NSP loss {print_nsp_loss:6.2f} | MLM loss {print_mlm_loss:6.2f}"

        self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)
        self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        return s

    def accuracy_summary(self, index, token, nsp, token_target, nsp_target, inverse_token_mask):
        """
        Create a summary of the accuracy

       Args: 
            index (int): Current batch index 
            token (Tensor): The token logits calculated by the model
            nsp (Tensor): The NSP logits calculated by the model
            token_target (Tensor): The target tokens for MLM
            nsp_target (Tensor): The target labels for NSP
            inverse_token_mask (Tensor): The inverse token mask to be used for MLM accuracy calculation
        Return: 
            String of the summary of the accuracy metrics 
        """
        global_step = self.current_epoch * len(self.loader) + index
        nsp_acc = nsp_accuracy(nsp, nsp_target)
        token_acc = token_accuracy(token, token_target, inverse_token_mask)

        self.writer.add_scalar("NSP train accuracy", nsp_acc, global_step=global_step)
        self.writer.add_scalar("Token train accuracy", token_acc, global_step=global_step)

        return f" | NSP accuracy {nsp_acc} | Token accuracy {token_acc}"

    def save_checkpoint(self, epoch, step, loss):
        """
        Save a checkpoint
        Args:
            epoch (int): The current epoch number.
            step (int): The current step number within the epoch.
            loss (float): The loss value at the current step.
        """
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.utcnow().timestamp():.0f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()

    def load_checkpoint(self, path: Path):
        """
        Load a checkpoint 

        Args:
            path (Path): The path to the checkpoint file to load.
        """
        print('=' * self._splitter_size)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
        print('=' * self._splitter_size)