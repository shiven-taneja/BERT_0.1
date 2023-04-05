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
import re

prepositions=["Using your cell phone while driving is [MASK] the law.",
              "[MASK] she's a little shy, she's a wonderful person once you get to know her.",
              "We drove [MASK] the coastline of California."
              ]
articles=['Carl lives alone in [MASK] one-bedroom apartment.',
              'The test results will be available in about [MASK] hour',
              'He is [MASK] best of what we have got.'
              ]

question_tags=["You would'nt like to invite my dad,[MASK] you?",
               'You do go to school,[MASK] you?',
               'They will wash the car,[MASK] they?'
               ]

opposites=['Black is the opposite of [MASK].',
           'Love is the opposite of [MASK].',
           'An atheist is the opposite of a [MASK].'
           ]

proverbs=['Better late than [MASK].',
          'Slow and [MASK] wins the race.',
          'When there is a [MASK] there is a way.'
          ]

#tokenizer and model for CHILDES data
tokenizer_childes = AutoTokenizer.from_pretrained(".\\trained_bert_mlm_childes_4_epoch")
model_childes = AutoModelForMaskedLM.from_pretrained(".\\trained_bert_mlm_childes_4_epoch")
 
#tokenizer and model for Child Text data
tokenizer_child_text = AutoTokenizer.from_pretrained(".\\trained_bert_mlm_child_text_4_epoch")
model_child_text = AutoModelForMaskedLM.from_pretrained(".\\trained_bert_mlm_child_text_4_epoch")

#tokenizer and model for Wiki Text data
tokenizer_wiki_text = AutoTokenizer.from_pretrained(".\\trained_bert_mlm_wiki_text_4_epoch")
model_wiki_text = AutoModelForMaskedLM.from_pretrained(".\\trained_bert_mlm_wiki_text_4_epoch")

unmasker_childes = pipeline('fill-mask', model=model_childes, tokenizer=tokenizer_childes)
unmasker_child_text = pipeline('fill-mask', model=model_child_text, tokenizer=tokenizer_child_text)
unmasker_wiki_text = pipeline('fill-mask', model=model_wiki_text, tokenizer=tokenizer_wiki_text)

print('Prepositions')
for i in prepositions:
  print(i)
  i=re.sub(r' {2,}',' ',i)
  print("BERT_CHILDES:"+str(unmasker_childes(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_childes(i)[0]['score']))
  print("BERT_CHILD_TEXT:"+str(unmasker_child_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_child_text(i)[0]['score']))
  print("BERT_WIKI_TEXT:"+str(unmasker_wiki_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_wiki_text(i)[0]['score']))
  print()
  print()

print('Articles')
for i in articles:
  print(i)
  i=re.sub(r' {2,}',' ',i)
  print("BERT_CHILDES:"+str(unmasker_childes(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_childes(i)[0]['score']))
  print("BERT_CHILD_TEXT:"+str(unmasker_child_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_child_text(i)[0]['score']))
  print("BERT_WIKI_TEXT:"+str(unmasker_wiki_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_wiki_text(i)[0]['score']))
  print()
  print()

print('Question Tags')
for i in question_tags:
  print(i)
  i=re.sub(r' {2,}',' ',i)
  print("BERT_CHILDES:"+str(unmasker_childes(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_childes(i)[0]['score']))
  print("BERT_CHILD_TEXT:"+str(unmasker_child_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_child_text(i)[0]['score']))
  print("BERT_WIKI_TEXT:"+str(unmasker_wiki_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_wiki_text(i)[0]['score']))
  print()
  print()

print('Opposites')
for i in opposites:
  print(i)
  i=re.sub(r' {2,}',' ',i)
  print("BERT_CHILDES:"+str(unmasker_childes(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_childes(i)[0]['score']))
  print("BERT_CHILD_TEXT:"+str(unmasker_child_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_child_text(i)[0]['score']))
  print("BERT_WIKI_TEXT:"+str(unmasker_wiki_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_wiki_text(i)[0]['score']))
  print()
  print()

print('Proverbs')
for i in proverbs:
  print(i)
  i=re.sub(r' {2,}',' ',i)
  print("BERT_CHILDES:"+str(unmasker_childes(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_childes(i)[0]['score']))
  print("BERT_CHILD_TEXT:"+str(unmasker_child_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_child_text(i)[0]['score']))
  print("BERT_WIKI_TEXT:"+str(unmasker_wiki_text(i)[0]['sequence'])+" CONFIDENCE:"+str(unmasker_wiki_text(i)[0]['score']))
  print()
  print()