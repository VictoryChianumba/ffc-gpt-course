import os
import lzma
from tqdm import tqdm

def xz_files_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xz') and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)

    return files

folder_path = 'openwebtext'
output_file = 'output{}.txt'
output_file_train = 'train_text.txt'
output_file_valid = 'valid_text.txt'
vocab_file = 'vocab.txt'

files = xz_files_dir(folder_path)
total_files = len(files)

# count the files 
# max_count = total_files // split_files if split_files != 0 else total_files

#  calculate split index's
split_index = int(total_files * 0.8)# split the files into 90-10 for training and testing
file_train = files[:split_index]
file_valid = files[split_index:]

#  process the files training and validation separately
vocab = set()

#  process the training files
with open(output_file_train, 'w', encoding='utf-8') as outfile:
    for  filename in tqdm(file_train, total = len(file_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

#  process the testing files
with open(output_file_valid, 'w', encoding='utf-8') as outfile:
    for  filename in tqdm(file_valid, total = len(file_valid)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)




with open(vocab_file, 'w', encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')