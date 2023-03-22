import csv, os, sys
import re

with open('train_output.csv', 'w', newline="", encoding="utf-8") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Paragraphs'])
    with open('wiki.test.tokens', 'r', encoding='utf-8') as txt:
        for line in txt.read().splitlines():
            if re.search("^[a-zA-Z]", line.strip()) is not None:
                csv_out.writerow(['test_tokens_file', line])
    csv_out.close()
    
with open('train_output_child.csv', 'w', newline="", encoding="utf-16") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Transcript'])
    with open('/Users/pri/COG403 Final Project/BERT_0.1/Baby Data/childes.txt', 'r') as txt:
        as_string = " ".join(line.strip() for line in txt)
        csv_out.writerow(['childes_data', as_string])
    csv_out.close()
