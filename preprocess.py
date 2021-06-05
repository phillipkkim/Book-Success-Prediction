""" PREPROCESS TEXTS AND SAVE TO CSV
"""

import os
import csv
from sklearn.model_selection import train_test_split


DATASET_HOME = "./novels/"

def preprocess():
    counter = 0
#         with open("data_val.csv", 'w') as csvfile_val:
    mydicts = []
    for f1 in os.listdir(DATASET_HOME):
            cand = DATASET_HOME + f1
            if os.path.isdir(cand):
                for f2 in os.listdir(cand):
                    cand2 = cand + "/" + f2
                    if os.path.isdir(cand2):
                        for f3 in os.listdir(cand2):
                            cand3 = cand2 + "/" + f3
                            if os.path.isdir(cand3):
                                for f4 in os.listdir(cand3):
                                    if f4.endswith(".txt"):
                                        filepath = cand3 + "/" + f4
                                        f = open(filepath, 'r')
                                        file_contents = f.read()
                                        
                                        label = 1 if filepath.find("success") != -1 else 0
                                        mydicts.append({"label": label, "content": ".".join(file_contents.split(".")[-10:])})

#                                         mydicts.append({"label": label, "content": " ".join(file_contents.split()[:10000])})
                                        f.close()
    #train, test = train_test_split(mydicts, test_size=0.2)                                
    with open("dataset/data_last10.csv", 'w') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = ["label", "content"])
        writer.writeheader()
        writer.writerows(mydicts)
    #with open("dataset/data_val_first100.csv", 'w') as csvfile: 
        #writer = csv.DictWriter(csvfile, fieldnames = ["label", "content"])
        #writer.writeheader()
        #writer.writerows(test)

preprocess()

