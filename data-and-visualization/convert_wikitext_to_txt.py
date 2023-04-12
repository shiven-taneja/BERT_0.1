import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

nltk.download('punkt')

input_file = 'wikitext-103\wiki.train.tokens'
output_file = 'wiki_train.txt'

# Function to clean and detokenize the text
def clean_and_detokenize(text):
    tokens = text.split() # Assuming tokens are space-separated
    detokenizer = TreebankWordDetokenizer()
    detokenized_text = detokenizer.detokenize(tokens)
    return detokenized_text

# Read the input file and write the detokenized content to the output file
with open(input_file, 'r', encoding='utf-8') as infile:
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            cleaned_line = clean_and_detokenize(line)
            outfile.write(cleaned_line + '\n')
