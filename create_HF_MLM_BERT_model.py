import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. Install Hugging Face Transformers library
# !pip install transformers

# 2. Prepare custom dataset
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
        self.examples = [tokenizer(line.strip(), truncation=True, max_length=512) for line in lines]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# 3. Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = CustomDataset("childes.txt", tokenizer)

# 4. Create a data loader
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 5. Load the BERT model architecture
config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=192 , num_hidden_layers=4, num_attention_heads=4, intermediate_size=768)
model = BertForMaskedLM(config=config)

# 6. Train the model
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs_childes_4_epoch",
    logging_steps=100,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# 7. Save the trained model
trainer.save_model("./trained_bert_mlm_childes_4_epoch")
tokenizer.save_pretrained("./trained_bert_mlm_childes_4_epoch")
