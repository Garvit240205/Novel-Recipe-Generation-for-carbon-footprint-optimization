# -*- coding: utf-8 -*-
"""Step1.py - Data Preprocessing for Recipe Database"""

import json
import numpy as np
import pandas as pd
import os
import re
import csv
import pickle
import shutil
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define file paths (adjusted for Ubuntu)
BASE_PATH = os.path.join('btp', 'btp')
BASE_PATH2 = os.path.join('RecipeDB-webdev', 'RecipeDB-webdev')
json_file_path = os.path.join(BASE_PATH2, 'RecipeDB_instructions.json')
local_json_copy = os.path.join(BASE_PATH, 'RecipeDB_instructions_copy.json')
local_csv_path = os.path.join(BASE_PATH, 'RecipeDB_instructions_copy.csv')
drive_csv_path = os.path.join(BASE_PATH, 'RecipeDB_instructions.csv')
ingredient_phrase_path = os.path.join(BASE_PATH2, 'RecipeDB_ingredient_phrase.csv')
general_file_path = os.path.join(BASE_PATH2, 'RecipeDB_general.csv')
ingredient_file_path = os.path.join(BASE_PATH2, 'RecipeDB_ingredient_phrase.csv')
instructions_file_path = os.path.join(BASE_PATH, 'RecipeDB_instructions.csv')
ingredient_details_path = os.path.join(BASE_PATH, 'Ingredient_details_server.csv')
output_dir = os.path.join(BASE_PATH, 'btp')
steps_path = os.path.join(BASE_PATH, 'data_v1.json')
titles_path = os.path.join(BASE_PATH, 'recipe_titles.csv')
ingredients_path = os.path.join(BASE_PATH, 'RecipeDB_general.csv')
output_pickle_path = os.path.join(BASE_PATH, 'data_v1.pickle')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Copy JSON file if needed and convert to CSV
if not os.path.exists(local_json_copy):
    shutil.copy(json_file_path, local_json_copy)
    print(f"Copied JSON file to: {local_json_copy}")

with open(local_json_copy, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

if isinstance(data, dict):
    data = [data]

if not data:
    raise ValueError("The JSON file is empty or not formatted correctly.")

headers = data[0].keys()
with open(local_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)
print(f"CSV file created at: {local_csv_path}")

# Step 2: Process ingredient phrases
df_phrase = pd.read_csv(ingredient_phrase_path, encoding='utf-8')

missing_units = df_phrase[df_phrase['unit'].isna() | (df_phrase['unit'].astype(str).str.strip() == '')]
print(f"Total rows with missing unit: {len(missing_units)}")

def fill_unit(group):
    non_missing = group['unit'].dropna().astype(str).str.strip()
    if not non_missing.empty:
        mode_unit = non_missing.mode().iloc[0]
        group['unit'] = group['unit'].fillna(mode_unit)
        group.loc[group['unit'].astype(str).str.strip() == '', 'unit'] = mode_unit
    else:
        group['unit'] = group['unit'].fillna("unknown")
    return group

df_phrase = df_phrase.groupby('ingredient', group_keys=False).apply(fill_unit)
missing_units_after = df_phrase[df_phrase['unit'].isna() | (df_phrase['unit'].astype(str).str.strip() == '')]
print(f"Total rows with missing unit after imputation: {len(missing_units_after)}")
print(df_phrase.head())

# Step 3: Merge data
df_general = pd.read_csv(general_file_path)
df_ingredient = pd.read_csv(ingredient_file_path)
df_instructions = pd.read_csv(instructions_file_path)

recipe_ids = df_general['Recipe_id'].unique()
df_filtered = df_ingredient[df_ingredient['recipe_no'].isin(recipe_ids)]
df_merged = pd.merge(
    df_filtered,
    df_general[['Recipe_id', 'Recipe_title', 'Continent', 'Region', 'Sub_region']],
    left_on='recipe_no',
    right_on='Recipe_id',
    how='inner'
).drop(columns=['Recipe_id'])

expected_columns = [
    'recipe_no', 'ingredient_Phrase', 'Recipe_title', 'Continent', 'Region', 'Sub_region',
    'ingredient', 'state', 'quantity', 'unit', 'temp', 'df', 'size', 'ing_id', 'ndb_id', 'M_or_A'
]
for col in expected_columns:
    if col not in df_merged.columns:
        df_merged[col] = pd.NA
df_final = df_merged[expected_columns]

output_file = os.path.join(output_dir, 'RecipeDB_general.csv')
df_final.to_csv(output_file, sep='\t', index=False)
print(f"New file written to: {output_file}")

# Step 4: Infer CO2 factors and weights
df_details = pd.read_csv(ingredient_details_path, encoding='latin1')

def normalize_name(name):
    if pd.isnull(name):
        return ""
    return re.sub(r'\s+', '', name.lower())

df_details['recipeDB_clean'] = df_details['RecipeDB Ingredient'].apply(normalize_name)
candidate_to_cf = {row['recipeDB_clean']: row['Carbon Footprint'] for _, row in df_details.iterrows() if row['recipeDB_clean']}
sorted_candidates = sorted(candidate_to_cf.keys(), key=len, reverse=True)

def infer_co2_factor(ingredient):
    ing_norm = normalize_name(ingredient)
    for candidate in sorted_candidates:
        if candidate in ing_norm:
            return candidate_to_cf[candidate]
    return np.nan

df_phrase['CO2_factor'] = df_phrase['ingredient'].apply(infer_co2_factor)

def approximate_weight(row):
    try:
        quantity = float(row['quantity'])
    except (TypeError, ValueError):
        quantity = None
    unit = str(row['unit']).lower().strip() if pd.notnull(row['unit']) else None
    size = str(row['size']).lower().strip() if pd.notnull(row['size']) else None
    ingredient = normalize_name(row['ingredient']) if pd.notnull(row['ingredient']) else None

    weight_units = {'kg': 1, 'g': 0.001, 'mg': 1e-6, 'lb': 0.453592, 'oz': 0.0283495}
    volume_units = {'cup': 0.24, 'cups': 0.24, 'teaspoon': 0.005, 'teaspoons': 0.005, 
                    'tablespoon': 0.015, 'tablespoons': 0.015, 'fluidounce': 0.03, 'fluidounces': 0.03, 'oz': 0.03}
    default_item_weight = {'water': 0.24, 'redlentil': 0.2, 'romtomato': 0.15, 'carrot': 0.07, 
                           'onion': {'small': 0.1, 'medium': 0.15, 'large': 0.2, 'default': 0.15}}

    if unit and quantity is not None:
        if unit in weight_units:
            return quantity * weight_units[unit]
        elif unit in volume_units:
            return quantity * volume_units[unit]
    if ingredient in default_item_weight:
        mapping_val = default_item_weight[ingredient]
        if isinstance(mapping_val, dict):
            return quantity * mapping_val.get(size, mapping_val['default']) if quantity else mapping_val.get(size, mapping_val['default'])
        return quantity * mapping_val if quantity else mapping_val
    return None

df_phrase['approx_weight_kg'] = df_phrase.apply(approximate_weight, axis=1)
df_phrase['CO2_emission'] = df_phrase.apply(
    lambda row: row['approx_weight_kg'] * row['CO2_factor'] if pd.notnull(row['approx_weight_kg']) and pd.notnull(row['CO2_factor']) else np.nan,
    axis=1
)

output_columns = ['recipe_no', 'ingredient_Phrase', 'ingredient', 'state', 'quantity', 'unit', 
                 'temp', 'df', 'size', 'ing_id', 'ndb_id', 'M_or_A', 'CO2_factor', 'approx_weight_kg', 'CO2_emission']
df_output = df_phrase[output_columns]
output_file = os.path.join(output_dir, 'RecipeDB_general.csv')
df_output.to_csv(output_file, sep='\t', index=False)
print(f"Output file written to: {output_file}")

# Step 5: Fetch unique recipe IDs
recipeIds = df_output['recipe_no'].unique().tolist()
recipeIdslistStringForm = [str(rid) for rid in recipeIds]
print(f"Number of Unique Recipe Ids: {len(recipeIdslistStringForm)}")

# Step 6: Filter JSON data
with open(local_json_copy, 'r', encoding='utf-8') as data_file:
    data = json.load(data_file)  # Corrected to use data_file

data = [item for item in data if str(item['recipe_id']) in recipeIdslistStringForm]
with open(steps_path, 'w', encoding='utf-8') as final:
    json.dump(data, final)
print(f"Filtered JSON saved to: {steps_path}")

# Step 7: Preprocessing
with open(steps_path, 'r', encoding='utf-8') as f:
    data_new = json.load(f)
print(f"Total Number of Recipes: {len(data_new)}")

def load_dataset(ingredients_path, steps_path, title_path):
    print("Loading all required files...")
    df_titles = pd.read_csv(title_path)
    ingredients = pd.read_csv(ingredients_path, sep='\t')
    with open(steps_path, 'r', encoding='utf-8') as json_file:
        steps = json.load(json_file)

    steps_dic = {int(dic['recipe_id']): {'instructions': dic['steps'] if isinstance(dic['steps'], list) else [dic['steps']]} for dic in steps}
    title_dic = {row['recipe_id']: row['Recipe_title'] for _, row in df_titles.iterrows()}
    continent_dict = {row['recipe_id']: row['Continent'] for _, row in df_titles.iterrows()}
    region_dict = {row['recipe_id']: row['Region'] for _, row in df_titles.iterrows()}
    sub_region_dict = {row['recipe_id']: row['Sub_region'] for _, row in df_titles.iterrows()}
    ingredient_dic = {rid: [] for rid in ingredients['recipe_no'].unique()}
    ing_phrase_dic = {rid: [] for rid in ingredients['recipe_no'].unique()}
    
    # Create a dictionary mapping ingredient to CO2_factor
    cf_dict = dict(zip(df_phrase['ingredient'], df_phrase['CO2_factor']))

    # Group by recipe_no to calculate total carbon footprint
    total_cf_per_recipe = ingredients.groupby('recipe_no')['CO2_emission'].sum().to_dict()

    dataset = []
    for rid in set(ingredients['recipe_no'].unique()):
        try:
            ingredient_list = ingredients[ingredients['recipe_no'] == rid]['ingredient'].tolist()
            recipe = {
                'ID': rid,
                'title': title_dic.get(rid),
                'ingredients': ingredient_list,
                'ingredient_phrase': ingredients[ingredients['recipe_no'] == rid]['ingredient_Phrase'].tolist(),
                'continent': continent_dict.get(rid),
                'region': region_dict.get(rid),
                'sub_region': sub_region_dict.get(rid),
                'instructions': steps_dic.get(rid, {'instructions': []})['instructions'],
                'carbon_footprint': total_cf_per_recipe.get(rid, np.nan),
                'ingredient_cf': [cf_dict.get(ing, np.nan) for ing in ingredient_list],
                'approx_weight_kg': ingredients[ingredients['recipe_no'] == rid]['approx_weight_kg'].tolist()
            }
            if recipe['title'] and recipe['instructions'] and recipe['ingredients']:
                dataset.append(recipe)
        except KeyError as e:
            print(f"Skipping recipe {rid} due to missing key: {e}")
    print("Dataset creation completed.")
    return dataset

# Generate recipe_titles.csv
df_titles = df_general[['Recipe_id', 'Recipe_title', 'Continent', 'Region', 'Sub_region']].rename(columns={'Recipe_id': 'recipe_id'})
df_titles.to_csv(titles_path, index=False)

# Run the dataset loading
data = load_dataset(ingredients_path, steps_path, titles_path)

# Preprocessing steps
def clean_ingredients(l):
    return list(set(lemmatizer.lemmatize(ele.lower()) for ele in l))

def fix_punctuation(l):
    return [re.sub(r'\s([?.!",](?:\s|$))', r'\1', instr) for instr in l]

p = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+)|(^\w*))')
def cap(match):
    return match.group().capitalize()

def fix_caps(l):
    return [p.sub(cap, instr.lstrip()) for instr in l]

for recipe in data:
    recipe['ingredients'] = clean_ingredients(recipe['ingredients'])
    recipe['instructions'] = fix_punctuation(recipe['instructions'])
    recipe['instructions'] = fix_caps(recipe['instructions'])
    for i, instr in enumerate(recipe['instructions']):
        if not instr.endswith('.'):
            recipe['instructions'][i] = instr.rstrip() + "."

# Save the final preprocessed data
with open(output_pickle_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Preprocessed data saved to: {output_pickle_path}")