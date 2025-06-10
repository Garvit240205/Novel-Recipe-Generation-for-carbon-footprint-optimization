import os
import torch
import pandas as pd
import numpy as np
import joblib as jb
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import csv
import math
import argparse

DENSITY_MAP = {
    'water': 1.0, 'milk': 1.03, 'oil': 0.92, 'flour': 0.57,
    'sugar': 0.85, 'salt': 1.2, 'butter': 0.91, 'honey': 1.42,
    'default': 1.0
}

VOLUME_TO_ML = {
    'ml': 1.0, 'milliliter': 1.0, 'milliliters': 1.0,
    'l': 1000.0, 'liter': 1000.0, 'liters': 1000.0,
    'tsp': 4.93, 'teaspoon': 4.93, 'teaspoons': 4.93,
    'tbsp': 14.79, 'tablespoon': 14.79, 'tablespoons': 14.79,
    'cup': 236.59, 'cups': 236.59,
    'oz': 29.57, 'ounce': 29.57, 'ounces': 29.57,
    'fl oz': 29.57, 'fluid ounce': 29.57, 'fluid ounces': 29.57,
}

MASS_TO_G = {
    'g': 1.0, 'gram': 1.0, 'grams': 1.0,
    'kg': 1000.0, 'kilogram': 1000.0, 'kilograms': 1000.0,
    'oz': 28.35, 'ounce': 28.35, 'ounces': 28.35,
    'lb': 453.59, 'lbs': 453.59, 'pound': 453.59, 'pounds': 453.59,
}

EACH_TO_G = {
    'egg': 55, 'eggs': 55,
    'clove': 5, 'cloves': 5,
    'pinch': 0.5, 'pinches': 0.5,
    'dash': 0.3, 'dashes': 0.3,
    'slice': 28, 'slices': 28,
    'onion': 150, 'onions': 150,
    'potato': 170, 'potatoes': 170,
    'carrot': 60, 'carrots': 60,
    'stalk': 40, 'stalks': 40,
    'sprig': 2, 'sprigs': 2,
    'head': 500, 'heads': 500,
    'can': 400, 'cans': 400,
}

def parse_fraction(qty_str):
    """
    Parses numbers, simple fractions (e.g., '1/2'), and mixed numbers (e.g., '1 1/2').
    Returns the decimal value or None if parsing fails.
    """
    qty_str = qty_str.strip()
    whole_part = 0.0
    fractional_part_str = qty_str

    if ' ' in qty_str:
        parts = qty_str.split(maxsplit=1)
        if len(parts) == 2:
            try:
                potential_whole = float(parts[0])
                if '/' in parts[1]:
                    whole_part = potential_whole
                    fractional_part_str = parts[1]
            except ValueError:
                pass

    value_from_fraction = None
    if '/' in fractional_part_str:
        frac_parts = fractional_part_str.split('/')
        if len(frac_parts) == 2:
            try:
                num = float(frac_parts[0].strip())
                den = float(frac_parts[1].strip())
                if den != 0:
                    value_from_fraction = num / den
                else:
                    print(f"Warning: Denominator is zero in fraction '{fractional_part_str}'")
            except ValueError:
                 pass

    if value_from_fraction is not None:
        return whole_part + value_from_fraction

    try:
        final_value = float(qty_str)
        return final_value
    except ValueError:
        return None

def convert_to_grams(quantity, unit, ingredient_name):
    """
    Converts quantity in various units to grams using approximations and defaults.
    Returns estimated grams (float) or 0.0 if conversion is impossible.
    """
    if quantity is None or quantity <= 0:
        return 0.0

    if unit is None:
        name_lower = ingredient_name.lower()
        matched_item_weight = None
        sorted_each_keys = sorted(EACH_TO_G.keys(), key=len, reverse=True)
        for item_unit in sorted_each_keys:
             if re.search(r'\b' + re.escape(item_unit) + r'\b', name_lower):
                 matched_item_weight = EACH_TO_G[item_unit]
                 break
        if matched_item_weight is not None:
             return quantity * matched_item_weight

        default_unitless_grams = 100.0
        print(f"Warning: Unitless ingredient '{ingredient_name}' not found in EACH_TO_G. Applying default approximation: {quantity} * {default_unitless_grams}g.")
        return quantity * default_unitless_grams

    unit_lower = unit.lower()

    if unit_lower in MASS_TO_G:
        return quantity * MASS_TO_G[unit_lower]

    if unit_lower in VOLUME_TO_ML:
        ml = quantity * VOLUME_TO_ML[unit_lower]
        density = DENSITY_MAP.get('default', 1.0)
        density_used = 'default'
        name_lower = ingredient_name.lower()
        sorted_density_keys = sorted(DENSITY_MAP.keys(), key=len, reverse=True)
        for key in sorted_density_keys:
             if key != 'default' and key in name_lower:
                 density = DENSITY_MAP[key]
                 density_used = key
                 break

        if density_used == 'default':
            print(f"Warning: No specific density found for '{ingredient_name}'. Using default density ({density:.2f} g/ml) for conversion from '{unit}'.")

        return ml * density

    if unit_lower in EACH_TO_G:
        return quantity * EACH_TO_G[unit_lower]

    print(f"Error: Unrecognized unit '{unit}' for ingredient '{ingredient_name}'. Cannot convert to grams. Setting grams to 0.0 for CF calculation.")
    return 0.0


def generate_recipe(model, tokenizer, device, input_ingredients):
    prompt = '<RECIPE_START> <INPUT_START> ' + ' <NEXT_INPUT> '.join(input_ingredients) + ' <INPUT_END>'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=768,
        temperature=1.0,
        top_k=30,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids('<RECIPE_END>')
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    return generated_text

def parse_recipe(text):
    title_start = text.find('<TITLE_START>') + len('<TITLE_START>')
    title_end = text.find('<TITLE_END>')
    title = text[title_start:title_end].strip() if title_start > -1 and title_end > -1 else "Generated Recipe"

    ingr_start = text.find('<INGR_START>') + len('<INGR_START>')
    ingr_end = text.find('<INGR_END>')
    ingr_text = text[ingr_start:ingr_end].strip() if ingr_start > -1 and ingr_end > -1 else ""
    ingredients = [i.strip() for i in ingr_text.split('<NEXT_INGR>')] if ingr_text else []

    instr_start = text.find('<INSTR_START>') + len('<INSTR_START>')
    instr_end = text.find('<INSTR_END>')
    instr_text = text[instr_start:instr_end].strip() if instr_start > -1 and instr_end > -1 else ""
    instructions = [i.strip() for i in instr_text.split('<NEXT_INSTR>')] if instr_text else []

    return title, ingredients, instructions

def extract_ingredients_and_quantities(ingredients_list):
    parsed_ingredients = []
    pattern = re.compile(
        r'^\s*'
        r'((?:[\d\.\/]+|\s)+?)'
        r'(?:\s*-\s*[\d\.\/]+(?:\s+[\d\/]+)?)?'
        r'\s+'
        r'([a-zA-Z]+(?:\(s\))?)?'
        r'\s*'
        r'(.*)'
        r'\s*$'
    )
    fallback_pattern = re.compile(r'^\s*(\d+\.?\d*(?:\s+\d+\s*\/\s*\d+)?)?\s*(.*)\s*$')

    all_units = set(list(VOLUME_TO_ML.keys()) + list(MASS_TO_G.keys()) + list(EACH_TO_G.keys()))

    for ing_str in ingredients_list:
        if not ing_str: continue
        ing_str = ing_str.strip()

        processed_ing_str = re.sub(r'(\d+)\s+\/\s*(\d+)', r'\1/\2', ing_str)

        match = pattern.match(processed_ing_str)
        qty = 1.0
        qty_str_parsed = "1"
        unit = None
        name = processed_ing_str
        original_name_before_clean = processed_ing_str

        if match:
            qty_str_match = match.group(1).strip()
            unit_candidate = match.group(2)
            name_candidate = match.group(3).strip()
            original_name_before_clean = name_candidate

            parsed_qty = parse_fraction(qty_str_match)

            if parsed_qty is not None:
                qty = parsed_qty
                qty_str_parsed = qty_str_match

                if unit_candidate and unit_candidate.lower() in all_units:
                    unit = unit_candidate
                    name = name_candidate
                else:
                    name = f"{unit_candidate} {name_candidate}".strip() if unit_candidate else name_candidate
                    unit = None

                if unit is None and name:
                    name_parts = name.split(maxsplit=1)
                    if len(name_parts) > 0 :
                        potential_unit = name_parts[0].lower()
                        if potential_unit in all_units:
                            unit = name_parts[0]
                            name = name_parts[1] if len(name_parts) > 1 else ""
            else:
                match = None
                name = processed_ing_str


        if not match:
            fallback_match = fallback_pattern.match(processed_ing_str)
            if fallback_match:
                 qty_str_fb = fallback_match.group(1)
                 name = fallback_match.group(2).strip()
                 original_name_before_clean = name

                 if qty_str_fb:
                     parsed_qty_fb = parse_fraction(qty_str_fb.strip())
                     if parsed_qty_fb is not None:
                         qty = parsed_qty_fb
                         qty_str_parsed = qty_str_fb.strip()
                     else:
                         qty = 1.0
                         qty_str_parsed = "1"
                         print(f"Warning: Could not parse fallback quantity '{qty_str_fb}' in '{processed_ing_str}'. Defaulting to 1.")
                 else:
                    qty = 1.0
                    qty_str_parsed = " "
                    unit = "unitless_qualitative"

                 if unit is None and name:
                     name_parts = name.split(maxsplit=1)
                     if len(name_parts) > 0:
                         potential_unit = name_parts[0].lower()
                         if potential_unit in all_units:
                             unit = name_parts[0]
                             name = name_parts[1] if len(name_parts) > 1 else ""

                 if name and qty == 1.0 and unit is None:
                     parts = name.split()
                     if len(parts) > 0:
                         last_word_lower = parts[-1].lower()
                         if last_word_lower in EACH_TO_G:
                             unit = parts[-1]
                             name = ' '.join(parts[:-1]).strip()
                         elif len(parts) > 1 and parts[0].lower() in ['large', 'medium', 'small'] and parts[1].lower() in EACH_TO_G:
                             unit = parts[1]
                             name = ' '.join(parts)


            else:
                name = processed_ing_str
                original_name_before_clean = name
                qty = 1.0
                qty_str_parsed = "1"
                unit = None
                print(f"Warning: Could not parse ingredient: '{processed_ing_str}'. Treating as '{name}' with quantity 1.")

        cleaned_name = name.strip().lower()
        common_suffixes = [
            'minced', 'chopped', 'diced', 'sliced', 'trimmed', 'cut into', 'washed', 'drained',
            'to taste', 'roughly chopped', 'finely chopped', 'quartered', 'optional', 'for garnish',
            'melted', 'softened', 'beaten', 'cooked', 'uncooked', 'peeled', 'seeded', 'cored',
            'pitted', 'rinsed', 'patted dry', 'at room temperature', 'divided', 'plus extra',
            'such as', 'about', 'approximately'
        ]
        suffix_pattern = r'\s*[,]?\s*(?:' + '|'.join(re.escape(s) for s in common_suffixes) + r')\b.*'
        cleaned_name = re.sub(suffix_pattern, '', cleaned_name, flags=re.IGNORECASE)
        cleaned_name = re.sub(r'\s*\([^)]*\)', '', cleaned_name)
        cleaned_name = re.sub(r'[,;:\-\.]\s*$', '', cleaned_name).strip()

        final_name = cleaned_name if cleaned_name else name.strip().lower()
        if not final_name:
             final_name = f"unknown_ingredient_{len(parsed_ingredients)}"
             print(f"Warning: Parsed empty ingredient name for '{processed_ing_str}'. Using default.")

        grams = 0.0
        if unit == "unitless_qualitative":
            grams = 1.0
        else:
            grams = convert_to_grams(qty, unit, final_name)
            if (grams is None or grams == 0.0) and final_name != original_name_before_clean.strip().lower():
                grams_retry = convert_to_grams(qty, unit, original_name_before_clean.strip().lower())
                if grams_retry is not None and grams_retry > 0:
                    grams = grams_retry

            grams = float(grams) if grams is not None else 0.0


        parsed_ingredients.append({
            'name': final_name,
            'original_qty': qty,
            'original_unit': unit,
            'grams': grams
        })

    return parsed_ingredients


def calculate_cf(parsed_ingredients, cf_dict, default_cf):
    total_cf = 0.0
    for ing_data in parsed_ingredients:
        norm_ing_base = re.sub(r'es$', '', ing_data['name']).strip()
        norm_ing_base = re.sub(r's$', '', norm_ing_base).strip()
        norm_ing = re.sub(r'\s+', '', norm_ing_base)

        cf_per_kg = cf_dict.get(norm_ing)
        match_type = "singular"
        if cf_per_kg is None:
             norm_ing_plural = re.sub(r'\s+', '', ing_data['name'])
             cf_per_kg = cf_dict.get(norm_ing_plural)
             match_type = "plural"
             if cf_per_kg is None:
                 cf_per_kg = default_cf
                 match_type = "default"

        qty_grams = ing_data['grams']

        if qty_grams > 0:
             cf_contribution = (qty_grams / 1000.0) * cf_per_kg
             total_cf += cf_contribution
    return total_cf

def adjust_quantities(parsed_ingredients, cf_dict, default_cf):
    if not parsed_ingredients:
        return []

    cf_values = []
    valid_indices = []
    ingredient_cfs = {}

    for i, ing_data in enumerate(parsed_ingredients):
         norm_ing_base = re.sub(r'es$', '', ing_data['name']).strip()
         norm_ing_base = re.sub(r's$', '', norm_ing_base).strip()
         norm_ing = re.sub(r'\s+', '', norm_ing_base)
         cf_per_kg = cf_dict.get(norm_ing)
         match_type = "singular"
         if cf_per_kg is None:
             norm_ing_plural = re.sub(r'\s+', '', ing_data['name'])
             cf_per_kg = cf_dict.get(norm_ing_plural)
             match_type = "plural"
             if cf_per_kg is None:
                 cf_per_kg = default_cf
                 match_type = "default"

         if ing_data['grams'] > 0 and cf_per_kg >= 0:
             cf_values.append(cf_per_kg)
             valid_indices.append(i)
             ingredient_cfs[i] = cf_per_kg

    if not cf_values:
         print("Warning: No ingredients with valid mass and non-negative CF found for adjustment.")
         adjusted_list = []
         for ing_data in parsed_ingredients:
             ing_data['adjusted_qty'] = ing_data['original_qty']
             ing_data['adjusted_grams'] = ing_data['grams']
             adjusted_list.append(ing_data)
         return adjusted_list

    c_max = max(cf_values) if cf_values else 1.0
    if c_max <= 0:
        print(f"Warning: Maximum Carbon Footprint (c_max) is not positive ({c_max:.4f}). Cannot apply relative adjustment. Ingredients will not be adjusted.")
        adjusted_list = []
        for ing_data in parsed_ingredients:
             ing_data['adjusted_qty'] = ing_data['original_qty']
             ing_data['adjusted_grams'] = ing_data['grams']
             adjusted_list.append(ing_data)
        return adjusted_list

    k_scaling_factor = 0.5

    adjusted_list = []
    for i, ing_data in enumerate(parsed_ingredients):
        if i in valid_indices:
            c_i = ingredient_cfs[i]

            adjustment_factor = 1.0 / (1.0 + k_scaling_factor * (c_i / c_max))

            original_qty_float = float(ing_data['original_qty']) if ing_data['original_qty'] is not None else 0.0
            new_qty = original_qty_float * adjustment_factor
            new_grams = ing_data['grams'] * adjustment_factor
        else:
            new_qty = ing_data['original_qty']
            new_grams = ing_data['grams']

        ing_data['adjusted_qty'] = new_qty
        ing_data['adjusted_grams'] = new_grams
        adjusted_list.append(ing_data)

    return adjusted_list


def format_ingredients(ingredients_data_list, use_adjusted=False):
    output_parts = []
    for ing_data in ingredients_data_list:
        qty_val = ing_data['adjusted_qty'] if use_adjusted else ing_data['original_qty']
        unit = ing_data['original_unit']
        name = ing_data['name']
        grams_to_display = ing_data['adjusted_grams'] if use_adjusted else ing_data['grams']

        if qty_val is None:
             qty_val = 0.0

        if abs(qty_val) < 1e-9:
             qty_str = "0"
        elif abs(qty_val - round(qty_val)) < 0.01:
            qty_str = str(int(round(qty_val)))
        elif abs(qty_val * 2 - round(qty_val * 2)) < 0.02:
             qty_str = f"{round(qty_val * 2) / 2:.1f}"
        elif abs(qty_val * 3 - round(qty_val * 3)) < 0.03:
             qty_str = f"{round(qty_val * 3) / 3:.2f}"
        elif abs(qty_val * 4 - round(qty_val * 4)) < 0.04:
             qty_str = f"{round(qty_val * 4) / 4:.2f}"
        else:
            qty_str = f"{qty_val:.2f}"

        gram_str_part = ""
        if grams_to_display is not None and grams_to_display > 0.1 and unit != "unitless_qualitative":
            gram_str_part = f" ({int(round(grams_to_display))}g)"

        if unit == "unitless_qualitative":
            output_parts.append(name)
            continue

        if unit:
            unit_str = unit
            is_plural = abs(qty_val - 1.0) > 0.01
            if is_plural and not unit.lower().endswith('s') and unit.lower() not in ['oz']:
                 if unit.lower().endswith(('h', 'x', 's', 'o', 'sh', 'ch', 'z')):
                     if unit.lower() in ['potato', 'tomato']:
                          unit_str += 'es'
                     elif not unit.lower().endswith('s'):
                          unit_str += 'es'
                     else:
                          unit_str += 's'
                 elif unit.lower() in ['pinch', 'dash', 'slice', 'sprig', 'box', 'bunch', 'glass', 'radish']:
                      unit_str += 'es' if unit.lower().endswith(('h', 'x', 's', 'sh', 'ch', 'z')) else 's'
                 elif len(unit) > 1 :
                      unit_str += 's'
            output_parts.append(f"{qty_str} {unit_str} {name}{gram_str_part}")
        else:
            output_parts.append(f"{qty_str} {name}{gram_str_part}")

    return ', '.join(output_parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and optimize recipes based on Carbon Footprint.")
    parser.add_argument("--model_dir", type=str, default="./outputs", help="Directory containing the pre-trained GPT-2 model and tokenizer.")
    parser.add_argument("--cf_data", type=str, default="./data/Ingredient_details_server.csv", help="Path to the Ingredient Carbon Footprint CSV file.")
    parser.add_argument("--recipe_data", type=str, default="./data/data_v1.pickle", help="Path to the input recipe data pickle file.")
    parser.add_argument("--output_csv", type=str, default="./results/generated_recipes_optimized.csv", help="Path to save the output CSV file.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of recipes to process (inclusive).")
    parser.add_argument("--end_index", type=int, default=-1, help="End index of recipes to process (exclusive). -1 processes till the end.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (-1 for CPU).")
    return parser.parse_args()

def main(args):

    if args.gpu_id >= 0 and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.gpu_id >= num_gpus:
            print(f"Error: GPU ID {args.gpu_id} is invalid. Available GPUs: {list(range(num_gpus))}. Falling back to GPU 0.")
            args.gpu_id = 0
            if num_gpus == 0:
                 print("Error: No GPUs available. Falling back to CPU.")
                 device = torch.device("cpu")
            else:
                 device = torch.device(f"cuda:{args.gpu_id}")
        else:
             device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: cuda:{args.gpu_id}")
    else:
        if not torch.cuda.is_available() and args.gpu_id >= 0:
             print("Warning: CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
        print("Using CPU")

    output_dir = args.model_dir
    print(f"Loading tokenizer from: {output_dir}")
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    print(f"Loading model from: {output_dir}")
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")

    print("Loading CF data...")
    try:
        cf_data = pd.read_csv(args.cf_data, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: CF data file not found at {args.cf_data}")
        return
    except Exception as e:
        print(f"Error loading CF data from {args.cf_data}: {e}")
        return

    cf_data['normalized_name'] = cf_data['RecipeDB Ingredient'].apply(lambda x: re.sub(r'\s+', '', str(x).lower()) if pd.notnull(x) else '')
    cf_dict = dict(zip(cf_data['normalized_name'], cf_data['Carbon Footprint']))
    default_cf = cf_data['Carbon Footprint'].median()
    print(f"CF data loaded. Default CF: {default_cf:.4f}")

    print("Loading recipe data...")
    try:
        df_new = jb.load(args.recipe_data)
    except FileNotFoundError:
        print(f"Error: Recipe data file not found at {args.recipe_data}")
        return
    except Exception as e:
        print(f"Error loading recipe data from {args.recipe_data}: {e}")
        return

    recipes = pd.DataFrame(df_new).to_dict(orient='records')
    print(f"Recipe data loaded. Total recipes: {len(recipes)}")

    output_dir_path = os.path.dirname(args.output_csv)
    if output_dir_path and not os.path.exists(output_dir_path):
        print(f"Creating output directory: {output_dir_path}")
        os.makedirs(output_dir_path)

    csv_path = args.output_csv
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    print(f"Output CSV will be saved to: {csv_path}")

    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(['Original Recipe ID', 'Generated Title', 'Original Ingredients', 'Original CF (kg CO2e)', 'Optimized Ingredients', 'Optimized CF (kg CO2e)', 'Instructions'])

            start = args.start_index
            end = args.end_index if args.end_index > 0 else len(recipes)
            if end > len(recipes):
                print(f"Warning: End index {args.end_index} is out of bounds ({len(recipes)} recipes). Processing up to the end.")
                end = len(recipes)
            if start >= end:
                print(f"Error: Start index {start} is not less than end index {end}. No recipes to process.")
                return

            print(f"Processing recipes from index {start} to {end-1}")

            for idx in range(start, end):
                print(f"\n--- Processing Recipe Index: {idx} ---")
                recipe = recipes[idx]
                original_id = recipe.get('ID', f'UnknownID_{idx}')
                input_ingredients = [str(ing) for ing in recipe.get('ingredients', [])]

                print(f"Generating recipe for ID: {original_id} with inputs: {input_ingredients}")
                if not input_ingredients:
                    print("Skipping recipe generation due to empty input ingredients.")
                    generated_text = ""
                    title = "Error: No Input Ingredients"
                    ingredients_list = []
                    instructions = []
                else:
                    generated_text = generate_recipe(model, tokenizer, device, input_ingredients)
                    title, ingredients_list, instructions = parse_recipe(generated_text)

                print(f"Generated Title: {title}")

                print("Extracting and Parsing Ingredients...")
                parsed_ingredients = extract_ingredients_and_quantities(ingredients_list)

                original_cf = calculate_cf(parsed_ingredients, cf_dict, default_cf)

                print("Adjusting Quantities (using scaled formula 1/(1 + k*ci/cmax) with k=0.5)...")
                adjusted_ingredients_data = adjust_quantities(parsed_ingredients, cf_dict, default_cf)

                temp_adjusted_for_cf_calc = [{'name': d['name'], 'grams': d['adjusted_grams']} for d in adjusted_ingredients_data]
                new_cf = calculate_cf(temp_adjusted_for_cf_calc, cf_dict, default_cf)

                original_ings_str = format_ingredients(parsed_ingredients, use_adjusted=False)
                optimized_ings_str = format_ingredients(adjusted_ingredients_data, use_adjusted=True)
                instructions_str = ' '.join(instructions)

                print(f"Original Recipe ID: {original_id}")
                print(f"Original Ingredients: {original_ings_str}")
                print(f"Original CF: {original_cf:.5f}")
                print(f"Optimized Ingredients: {optimized_ings_str}")
                print(f"Optimized CF: {new_cf:.5f}")
                print(f"Instructions: {instructions_str}")
                print("---")

                writer.writerow([original_id, title, original_ings_str, original_cf, optimized_ings_str, new_cf, instructions_str])

            print(f"\nRecipe generation and post-processing complete. Data saved to {csv_path}.")

    except IOError as e:
        print(f"Error writing to CSV file {csv_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)