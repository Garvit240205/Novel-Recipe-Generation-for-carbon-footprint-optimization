# Novel Recipe Generation for Carbon Footprint Optimization

## Overview

This project focuses on the development and evaluation of a system for generating culinary recipes with an optimized (reduced) carbon footprint (CF). The core idea is to leverage a pre-trained language model (GPT-2) for recipe generation and then apply a novel post-processing algorithm to adjust ingredient quantities, thereby minimizing the recipe's overall environmental impact without drastically altering its nature.

## Key Accomplishments

### 1. Advanced Ingredient Parsing and Quantification:
*   Developed a robust ingredient parser capable of extracting quantities, units, and ingredient names from free-form text generated by a language model.
*   Successfully implemented logic to handle various quantity formats, including whole numbers, decimals, simple fractions (e.g., "1/2"), and mixed numbers (e.g., "1 1/2", "0.5 1/2").
*   Integrated a normalization step to handle spaced fractions (e.g., "1 /2" correctly interpreted as "1/2").
*   Implemented comprehensive unit conversion logic, converting diverse volume (cups, tbsp, ml, etc.) and mass (g, kg, oz, lb) units into a standardized gram measurement for accurate CF calculation. This includes approximate density conversions for volume-to-mass calculations.
*   Incorporated a system for estimating gram weights for unitless or "each" type ingredients (e.g., "1 onion", "2 eggs") based on common approximations.

### 2. Carbon Footprint Calculation and Data Integration:
*   Successfully integrated an external dataset providing carbon footprint values (kg CO2e per kg of ingredient) for a wide range of food items.
*   Developed a matching system to link parsed ingredient names (after normalization and cleaning) to their corresponding CF values in the database, including a fallback to a default CF value for unlisted ingredients.
*   Implemented a precise CF calculation for each generated recipe based on the determined gram weights of its ingredients and their respective CF values.

### 3. Novel CF Optimization Algorithm:
*   Designed and implemented an algorithm to adjust ingredient quantities in a generated recipe to reduce its total carbon footprint.
*   The optimization uses a scaled adjustment factor (1 / (1 + k * c_i / c_max)) where `k` is a scaling parameter (set to 0.5), `c_i` is the CF of the ingredient, and `c_max` is the maximum CF among ingredients in the recipe. This approach aims to reduce high-CF ingredients more significantly while proportionally reducing others.
*   The system calculates both the original CF and the optimized CF, allowing for direct comparison and evaluation of the reduction.

### 4. End-to-End Recipe Generation and Optimization Pipeline:
*   Established a complete pipeline that takes a list of input ingredients, uses a fine-tuned GPT-2 model to generate a full recipe (title, ingredients, instructions), parses the generated ingredients, calculates the original CF, applies the optimization algorithm, calculates the new CF, and formats both original and optimized recipes for output.
*   The pipeline is designed for batch processing, capable of iterating through a dataset of input ingredient lists and generating/optimizing recipes for each.

### 5. Comprehensive Data Analysis and Validation:
*   Performed detailed data analysis on a batch of generated and optimized recipes to quantify the effectiveness of the CF reduction.
*   Generated key performance metrics, including average original CF, average optimized CF, average absolute reduction, and average percentage reduction.
*   Produced a suite of visualizations (histograms, scatter plots, bar charts) clearly demonstrating the shift towards lower CF values, the distribution of reductions, and the overall success of the optimization strategy.

### 6. Robust Output and Reporting:
*   Structured the output into a CSV format, capturing essential details for each recipe: Original Recipe ID, Generated Title, Original Ingredients string, Original CF, Optimized Ingredients string, Optimized CF, and Instructions.
*   The analysis script can process this CSV to produce summary statistics and visual plots, facilitating clear reporting of results.

### 7. Technical Implementation:
*   Leveraged Python with key libraries such as `transformers` (for GPT-2), `pandas` (for data manipulation), `numpy` (for numerical operations), `matplotlib` and `seaborn` (for plotting), and `re` (for text parsing).
*   The system includes error handling, sensible defaults for missing data (e.g., default CF, default densities), and warnings for unparseable or ambiguous inputs.

This project successfully demonstrates a viable approach to integrating environmental considerations directly into the automated recipe generation process, paving the way for tools that can help users make more sustainable food choices.
