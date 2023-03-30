import csv, os, sys
import re
import pandas as pd

with open('train_output_child.csv', 'w', newline="", encoding="utf-8") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Transcript'])
    with open('childes.txt', 'r') as txt:
        as_string = " ".join(line.strip() for line in txt)
        csv_out.writerow(['childes_data', as_string])
        
    df = pd.read_csv('children_books.csv')
    for index, row in df.iterrows():
        csv_out.writerow(['children_books', row['Desc']])
        
    df2 = pd.read_csv('children_stories.csv')
    for index, row in df2.iterrows():
        csv_out.writerow(['children_stories', row['desc']])
                        
    csv_out.close()
    
with open('train_output.csv', 'w', newline="", encoding="utf-8") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Paragraphs'])
    with open('wiki.test.tokens', 'r', encoding='utf-8') as txt:
        for line in txt.read().splitlines():
            if re.search("^[a-zA-Z]", line.strip()) is not None:
                csv_out.writerow(['test_tokens_file', line])
    csv_out.close()
