#TRAIN / DEV / TEST SPLIT (70/15/15) FOR BIO DATASET
#Splitting the BIO-formatted address dataset into training, development/validation, and test sets
#using a 70/15/15 ratio.
#Each address is treated as a sequence
#-----------------------------------------------------------------------------

#Importing libraries
import random
from pathlib import Path

#Setting random seed for reproducibility
random.seed(42)

#Path to the BIO dataset
input_file = r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\addresses_bio.txt"
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read().strip()

#Split into individual address sequences (separated by blank lines)
sequences = content.split("\n\n")
print("Total addresses:", len(sequences))

#Shuffling sequences before splitting to avoid ordering bias
random.shuffle(sequences)

#70/15/15 split
train = sequences[:2100]
dev = sequences[2100:2550]
test = sequences[2550:]

#Output directory
from pathlib import Path
Path(r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs").mkdir(exist_ok=True)

#Write training set
with open(r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(train) + "\n")

#Write development/validation set
with open(r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\dev.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(dev) + "\n")

#Write test set
with open(r"C:\Users\valen\Desktop\MSc Dissertation\Code\FINAL\Outputs\test.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(test) + "\n")

print("Train:", len(train))
print("Dev:", len(dev))
print("Test:", len(test))