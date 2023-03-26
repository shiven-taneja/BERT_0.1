import csv, os, sys
import re

with open('train_output_child.csv', 'w', newline="", encoding="utf-8") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Transcript'])
    with open('childes.txt', 'r') as txt:
        as_string = " ".join(line.strip() for line in txt)
        csv_out.writerow(['childes_data', as_string])
    csv_out.close()
    
with open('train_output_child.csv', 'w', newline="", encoding="utf-16") as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['filename', 'Transcript'])
    with open('childes.txt', 'r') as txt:
        as_string = " ".join(line.strip() for line in txt)
        csv_out.writerow(['childes_data', as_string])
    csv_out.close()
