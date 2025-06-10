# -*- coding: utf-8 -*-
"""Step_2_Tokenization.py

Adapted for running on V100-server2 server with paths matching Step1.py.
"""

import re
import numpy as np
import pandas as pd
import joblib as jb
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import os
import h5py

# Define file paths matching Step1.py
BASE_PATH = os.path.join('btp', 'btp')
ROOT_DIR = "/home/garvit22185/Python-3.8.18/Python-3.8.18"
FULL_BASE_PATH = os.path.join(ROOT_DIR, BASE_PATH)

# Input and output file paths
input_pickle_path = os.path.join(FULL_BASE_PATH, "data_v1.pickle")
train_temp_path = os.path.join(FULL_BASE_PATH, "train_temp.txt")
test_temp_path = os.path.join(FULL_BASE_PATH, "test_temp.txt")
output_h5_path = os.path.join(FULL_BASE_PATH, "data_temp.h5")

# Load the updated recipe data from Step1 output
df_new = jb.load(input_pickle_path)
df = pd.DataFrame(df_new)
print("DataFrame head:")
print(df.head())

# Function to parse and optimize ingredient phrases with exponential reduction
def optimize_ingredient_phrases(ingredient_phrases, ingredient_cf):
    if not ingredient_cf or all(np.isnan(cf) for cf in ingredient_cf):
        return ingredient_phrases
    c_max = max([cf for cf in ingredient_cf if not np.isnan(cf)], default=0)
    if c_max == 0:
        return ingredient_phrases
    optimized_phrases = []
    for phrase, cf in zip(ingredient_phrases, ingredient_cf):
        if np.isnan(cf):
            optimized_phrases.append(phrase)
            continue
        match = re.match(r'(\d*\.?\d+)\s*(\w+)\s*(.*)', phrase.strip())
        if match:
            quantity, unit, ingredient = match.groups()
            try:
                quantity = float(quantity)
                reduction_factor = 1 / (1 + cf / c_max)
                optimized_quantity = quantity * reduction_factor
                optimized_phrase = f"{optimized_quantity:.2f} {unit} {ingredient}"
                optimized_phrases.append(optimized_phrase)
            except ValueError:
                optimized_phrases.append(phrase)
        else:
            optimized_phrases.append(phrase)
    return optimized_phrases

# Optimize ingredient quantities
df['ingredient_phrase'] = df.apply(lambda row: optimize_ingredient_phrases(row['ingredient_phrase'], row['ingredient_cf']), axis=1)

# Calculate new carbon footprint based on optimized quantities
def calculate_new_cf(row):
    if not row['ingredient_cf'] or all(np.isnan(cf) for cf in row['ingredient_cf']):
        return np.nan
    c_max = max([cf for cf in row['ingredient_cf'] if not np.isnan(cf)], default=0)
    if c_max == 0:
        return sum([w * cf for w, cf in zip(row['approx_weight_kg'], row['ingredient_cf']) if not np.isnan(w) and not np.isnan(cf)])
    reduction_factors = [1 / (1 + cf / c_max) if not np.isnan(cf) else 1 for cf in row['ingredient_cf']]
    new_co2_emissions = [w * cf * rf for w, cf, rf in zip(row['approx_weight_kg'], row['ingredient_cf'], reduction_factors) if not np.isnan(w) and not np.isnan(cf)]
    return sum(new_co2_emissions) if new_co2_emissions else np.nan

df['carbon_footprint'] = df.apply(calculate_new_cf, axis=1)

# Process instructions
list_of_instrns = []
for row in range(len(df)):
    instr = df.iloc[row]['instructions']
    strg = ""
    length = len(instr) - 1
    count = 0
    for instruction in instr:
        processed_instr = []
        for j in range(len(instruction)):
            if instruction[j] in ['|', '\t']:
                continue
            elif instruction[j] == ' ':
                if j > 0 and instruction[j-1] != '|':
                    strg += instruction[j]
            elif instruction[j] == '.' and j != len(instruction)-1 and not (j > 0 and instruction[j-1].isdigit()):
                strg += ' ' + instruction[j]
            elif instruction[j].isalpha():
                strg += instruction[j].lower()
            elif instruction[j] == ',':
                strg += ' ' + ','
            elif instruction[j].isdigit():
                if j+1 < len(instruction) and instruction[j+1] == '.' or j+2 < len(instruction) and instruction[j+2] == '.':
                    continue
                else:
                    strg += instruction[j]
        if count != length:
            strg += ' ; '
        count += 1
    processed_instr.append(strg)
    list_of_instrns.append(processed_instr)

df.drop('instructions', inplace=True, axis=1)
df['instructions'] = list_of_instrns
print("Updated DataFrame head:")
print(df.head())

# Split into train and test sets
train, test = train_test_split(df, train_size=0.96, random_state=2)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print("Train Portion size is:", train.shape)
print("Test Portion size is:", test.shape)

# Function to convert DataFrame to plaintext format
def df_to_plaintext_file(input_df, output_file):
    print("Writing to", output_file)
    with open(output_file, 'w', encoding="utf-8") as f:
        for index, row in input_df.iterrows():
            title = row.title
            instructions = row.instructions[0].split('.')[:-1]
            ingredients = row.ingredient_phrase
            keyword = row.ingredients
            cf = row['carbon_footprint']

            if index % 40000 == 0:
                print(index)
                print("ingreds --->", ingredients)
                print("keywords --->", keyword)
                print("cf --->", cf)

            res = "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(keyword) + " <INPUT_END> <TITLE_START> " + \
                  title + "<TITLE_END> <INGR_START> " + " <NEXT_INGR> ".join(ingredients) + " <INGR_END> " + \
                  "<CF_START> " + str(cf) + " <CF_END> <INSTR_START> " + " <NEXT_INSTR> ".join(instructions) + \
                  " <INSTR_END> <RECIPE_END>"
            f.write("{}\n".format(res))

# Save processed train and test files
df_to_plaintext_file(train, train_temp_path)
df_to_plaintext_file(test, test_temp_path)

# Initialize GPT-2 Tokenizer and add special tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
special_tokens = {
    "additional_special_tokens": ['<RECIPE_START>', '<INPUT_START>', '<NEXT_INPUT>', '<INPUT_END>',
                                  '<INGR_START>', '<NEXT_INGR>', '<INGR_END>', '<INSTR_START>',
                                  '<NEXT_INSTR>', '<INSTR_END>', '<TITLE_START>', '<TITLE_END>',
                                  '<RECIPE_END>', '<CF_START>', '<CF_END>']
}
tokenizer.add_special_tokens(special_tokens)
end_token_id = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]

# Tokenize and save to HDF5 file
hf = h5py.File(output_h5_path, "w")
for filename in [test_temp_path, train_temp_path]:
    out_np = []
    data = open(filename, "r")
    num = 0
    rows = 0
    last = []
    for line in data:
        num += 1
        if num % 10000 == 0:
            print("Read " + str(num) + " Written: " + str(rows))

        text_tokens = tokenizer.tokenize(line)
        if len(text_tokens) > 1024:
            continue

        text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        if (len(last) + len(text_tokens_ids)) <= 1024:
            last += text_tokens_ids
        else:
            while len(last) < 1024:
                last.append(end_token_id)
            out_np.append(last)
            last = text_tokens_ids
            rows += 1
    out_mat = np.matrix(out_np)
    print(out_mat.shape)
    hf.create_dataset(filename, data=out_mat)
hf.close()

print("Final length of tokenizer:", len(tokenizer))

t = []
with open(train_temp_path) as file1:
    for f in file1:
        t.append(f)
print('No of recipes downsampled for prototyping:', len(t))