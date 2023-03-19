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
