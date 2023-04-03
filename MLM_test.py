import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup,
    DefaultDataCollator,
    TrainingArguments, 
    Trainer,
    pipeline
)
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load tokenizer and models babyberta
# tokenizer = AutoTokenizer.from_pretrained("phueb/BabyBERTa-3")
# model = AutoModelForMaskedLM.from_pretrained("phueb/BabyBERTa-3")

#bert
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

#Our own ModelForMaskedLM Model 
tokenizer = AutoTokenizer.from_pretrained(".\\trained_bert_mlm_child_text_4_epoch")
model = AutoModelForMaskedLM.from_pretrained(".\\trained_bert_mlm_child_text_4_epoch")

unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
print(unmasker("The Milky Way is a [MASK] galaxy."))
