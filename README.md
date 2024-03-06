<div align="center" width="50">

<img src="./Files Used in Readme/Colorful Mutton Biryani YouTube Thumbnail.gif" alt="Welcome!" height="500" width="1000"/>

</div>

## ⚡ Introduction

Due to the abundance of cooking recipes available online, there is growing interest in harnessing this
information to develop new recipes. In order to solve the "Novel Recipe Generation" problem, our
primary goal is to create new, realistic cooking recipes that are so similar to the real ones that even
culinary experts can't tell the difference. We fed a vast amount of structured recipe data into a deep
learning model like GPT-2 in order to create these innovative recipes.

## 📝 Background Study

We used the GPT2 model to generate novel recipes. GPT2 is essentially a machine learning
language model that can anticipate the following word from a sentence's context (single token).
As part of their investigation, the Open 11 researchers searched the internet to gather a sizable
40GB dataset known as Web Text, which was used to train the GPT-2. The trained GPT-2's smallest
variant requires 500 MB of storage to keep all of its parameters. Given that the largest GPT-2 model
is 13 times larger, it may use more than 6.5 GB of storage. Transformer decoding blocks are used to
construct GPT2. GPT2 outputs one token at a time. GPT2 is auto-regressive in nature; each token in
the sentence has the context of the previous words. After each token is produced, that token is
added to the sequence of inputs. And that new sequence becomes the input to the model in its next
step. This is an idea called "auto-regression". The GPT-2 model can take into account sampling
words other than the top word by using a parameter called top-k, which is the case when top-k = 1.
While the GPT 3 model is trained on 175 billion parameters, the GPT2 model uses about 1.5 billion
parameters.

Overview of feeding in text and generating a single token in GPT2:
1. Tokenization - Take some words, break them up into their common pieces. Take those
common pieces and replace them with a number. Tokenization is necessary because
computers only work with numbers. This also represents words efficiently.
2. Embedding with time signal - Take that 1 string of numbers and convert each number to a
vector. This captures the position of words relative to one another and allows words to take
value from other words associated with them, e.g. "The boy ran through the woods, and he
surely had not stolen the cherry pie for which they were chasing him." In this sentence "he"
should clearly tie a lot of its meaning to "boy".
3. Decoder Block - The pieces are self-attention blocks, feedforward neural nets, and
Normalization. Self-attention blocks identify which words to focus on. In the sentence, "Jimmy
played with the burning bush, and then went around to the next bush," the words "Jimmy,"
"played," "burning," and "bush" capture a high proportion of the meaning in that sentence. This
idea that certain words and phases capture more meaning and thus should be given more
"attention" is the intuition of self-attention blocks.
4. Linear Layer - Prior to the tokenization process, a vocabulary size will be decided upon and a
vocabulary will be set up.
5. The vocabulary is just a list of all the possible tokens (numbers) that can be produced and
which letter or group of letters the tokens are equal to. The linear layer takes the output of the
last decoder block and converts it to a vector whose dimensions are vocabulary size by 1. In
short, it takes a lot of inputs and produces a list where each spot represents a token. The
higher the number in the spot the better the chance that that token is the best pick.
6. Softmax - Converts the output of the linear layer to a probability distribution. The output of the
linear layer tells you information about which tokens are the best picks, but it is hard to use.
The values range from very small values(huge negative values) to very large values and their
meaning is in relation to all the other values. To make them easier to use apply the Softmax
function which converts the vector to a probability distribution. This means each number
represents the probability that that token is the correct one.
7. Pick a token - Choose the method to pick the next token from the probability distribution of
tokens, and use that method to pick the token. There are various methods to do so including,
greedy, temperature sampling, nucleus sampling, and top-k sampling. Convert Token to a
word piece using the vocabulary.

## <div align="center">:star: Methodology</div>
## 📜 Dataset:
We used the RecipeDB_v1 dataset which is an organized collection of recipes, ingredients, and
nutrition profiles related to flavor profiles and health associations. The repertoire consists of the
laborious fusion of over 1,18,000 recipes from cuisines around the world (six continents, 26
geo-cultural regions, and 74 countries), cooked with over 23,500 ingredients from various categories
using 268 different processes (heat, cook, boil, simmer, bake, etc.), and further linked to their flavor
molecules (FlavorDB), nutritional profiles (USDA), and empirical records of disease associations
obtained from Medline (DietRx).
We used the file “Recipe_correct_ndb_updated_v1.csv” in the RecipeDB dataset to go ahead with
our problem. The RecipeDB dataset consists of 1154404 records. 

Below are the 12 fields in the dataset:
<br>
<table>
<tbody>
 <tr>
<td align="center" width="20%">
<span><b><center>recipe_no</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>state</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>size</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>ingredient_Phrase</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>quantity</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>unit</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>temp</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>df</center></b></span> 
</td>



<td align="center" width="20%">
<span><b><center>ingredient</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>ing_id</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>ndb_id</center></b></span> 

</td>

<td align="center" width="20%">
<span><b><center>M_or_A</center></b></span> 
</td>
</tr>

</tbody>
</table>

Out of all the fields, recipe_no, ingredient fields are used further to
calculate the popularity of the ingredients across the recipes.


## <div align="center">:star: Methods/Workflow</div>
## 🔍 Data Pre-Processing: 
Following steps were performed for the data preprocessing:
1. Removal of recipes: Recipes which contain only a single ingredient are removed from the
recipe database because they are not of much importance.
2. Merging of useful Information corresponding to recipes: Recipes are mapped with their
corresponding instructions, region, continent, sub-region.
3. Lemmatization: Performed lemmatization on the ingredients set of all the recipes to convert all
forms of each ingredient to their lemmatized form. Before performing this, all ingredient words
are converted to lowercase so that the model should be able to analyze the same ingredients
present in multiple recipes because we want that “potato” in one recipe and “Potato” in
another recipe should be considered the same by the model.
4. Fixing Punctuations: This operation is also performed on the ingredients set of all recipes.
5. Removing Useless Symbols: Some symbols are present in recipe instructions that are not of
any use, therefore we have removed all of these symbols from the instructions of all the
recipes.

## 🌐 Tokenization:
Along with the default tokens used by the GPT2 model, we have used the following tokens to
differentiate different parts of the recipe data.
<div align="center">
<br>
<table>
<tbody>
 <tr>
<td align="center" width="20%">
<span><b><center>TOKEN</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>DEFINITION</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>BEGIN_RECIPE</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Start of the recipe</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>BEGIN_INPUT</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Start of the input ingredient given</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>NEXT_INPUT</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Next ingredient in the input ingredients</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>END_INPUT</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>End of the input ingredients</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>BEGIN_TITLE</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Start of the recipe title</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>END_TITLE</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>End of the recipe title</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>BEGIN_INGREDS</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Start of the recipe ingredients</center></b></span> 
</td>
</tr>


<tr>
<td align="center" width="20%">
<span><b><center>NEXT_INGREDS</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Next ingredient in the ingredient list</center></b></span> 
</td>
</tr>


<tr>
<td align="center" width="20%">
<span><b><center>END_INGREDS</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>End of the recipe ingredients</center></b></span> 
</td>
</tr>


<tr>
<td align="center" width="20%">
<span><b><center>BEGIN_INSTR</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Start of the recipe instructions</center></b></span> 
</td>
</tr>


<tr>
<td align="center" width="20%">
<span><b><center>NEXT_INSTR</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Next instructions in the instructions</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>END_INSTR</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>End of the recipe instructions</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>END_RECIPE</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>End of the recipe</center></b></span> 
</td>
</tr>
</table>
</div>

## 💻 Model Training
We have used the 118083 recipes data. Initially, we splitted the data into train and test sets where the
train set contains 94.6 % and the test set contains 5.4% of the data. Then we attach the tokens
defined above to each of the sets, as a result, train.txt and test.text files are generated. Then the
model is trained using the recipes of the train data. 

Following values of functional parameters of GPT 2 model are used to train the model:
<br>
<table>
<tbody>
 <tr>
<td align="center" width="20%">
<span><b><center>num_train_epochs=2,
per_device_train_batch_size=4,
per_device_eval_batch_size=4,
gradient_accumulation_steps=8,
evaluation_strategy="steps",
fp16=True,
fp16_opt_level='O1',
warmup_steps=1e2,
learning_rate=5e-4,
adam_epsilon=1e-8,
weight_decay=0.01,
save_total_limit=1,
load_best_model_at_end=True
</center></b></span> 
</td>
</tr>
</table>

Model training had taken about 1 to 1.5 hours.

## ⚡ Generation of Recipes
We used the RecipeDB data to train the GPT2 model, which we then used to produce new recipes.
The list of ingredients was submitted to the GPT2 model. The method used to create the list of
ingredients is listed below.
1. A random number from 2 and 8 was generated, and that many ingredients were chosen at random
in proportion to how frequently they appeared across the recipes. The list of those ingredients was
fed into the GPT2 model.
<br>
<table>
<tbody>
 <tr>
<td align="center" width="20%">
<span><b><center>RANDOM INGREDIENTS</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>INGREDIENT PHRASES</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>RECIPE TITLE</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>RECIPE INSTRUCTIONS</center></b></span> 
</td>
</tr>

<tr>
<td align="center" width="20%">
<span><b><center>kalamata olive,onion soup mix,romaine lettuce leaf,chocolate,beef bouillon granule,shiitake mushroom,yogurt</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>1 cup plain yogurt | 1/4 cup grated miniature semisweet chocolate ( such as kittencal's ) | 3 tablespoons dried onions soup mix | 1 1/2 teaspoons beef bouillon granules | 8 -10 kalamata olives, rinsed and drained | 2 packages sliced shiitake mushrooms, stems removed ( about 2 cups ) | 1 small head romaine lettuce leaf</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>Korean Beef and Shiitake Lettuce Wraps</center></b></span> 
</td>

<td align="center" width="20%">
<span><b><center>In a bowl, stir together the yogurt, chocolate, onion soup mix, and beef bouillon granules. Set aside. To make slaw, in a bowl, stir together the olives, mushrooms, lettuce, and 1/3 the reserved yogurt mixture until evenly mixed. Top each wrap with the remaining yogurt mixture, spreading evenly. Serve immediately.</center></b></span> 
</td>
</tr>
</table>

## 🎯 Results
<div align="center" width="50">

<img src="./Files Used in Readme/resultsStatistics.gif" alt="Welcome!" height="400" width="800"/>

</div>

## ⚡ Languages & Tools Used
<p align="center">
 <img src="http://img.shields.io/badge/-Git-F1502F?style=flat&logo=git&logoColor=FFFFFF" height="32">
 <img src="https://github.com/anishghimire603/anishghimire603/blob/master/Assets/python.svg" alt="python" style="vertical-align:top; margin:4px">
 <img src="https://github.com/anishghimire603/anishghimire603/blob/master/Assets/ai.svg" alt="ai" style="vertical-align:top; margin:4px">
 <img src="https://github.com/anishghimire603/anishghimire603/blob/master/Assets/datascience.svg" alt="datascience" style="vertical-align:top; margin:4px">
 <img src="http://img.shields.io/badge/-Github-000000?style=flat&logo=github&logoColor=FFFFFF" height="32">
 <img src="https://github.com/anishghimire603/anishghimire603/blob/master/Assets/visualstudio_code.svg" alt="vscode" style="vertical-align:top; margin:4px">
 <img src="https://raw.githubusercontent.com/8bithemant/8bithemant/master/svg/dev/misc/chrome.svg" alt="Twitter" style="vertical-align:top; margin:4px">
 <img src="https://github.com/anishghimire603/anishghimire603/blob/master/Assets/jetbrains_pycharm.svg" alt="pycharm" style="vertical-align:top; margin:4px">
 <img src="https://raw.githubusercontent.com/alexnaiman/alexnaiman/master/resources/dev/xcode.svg" alt="xcode" style="vertical-align:top; margin:4px">
 
</p>

## 📊 Conclusion
Using the trained GPT2 model, we produced 1,20,000 recipes in total. The RecipeDB data was first processed, then special tokens were added, and finally the train set and test set were separated. The GPT2 model, which had been trained on the train set, was fed a list of random ingredients, and it then developed a recipe using those items plus a few extras.

## ⚡Future Work
Ratatouille 3.0 implementation is based on popularity constraint. More constraints can be used in the future versions of Ratatouille like: cuisine, dish type, nutrition, cost, ingredient availability, allergy, health condition, dietary preferences.

## ⚡References
[Ratatouille: A tool for Novel Recipe Generation, IEEE 38th International Conference on Data Engineering Workshops (ICDEW) (2022).](https://ieeexplore.ieee.org/document/9814641)

## ⚡Developers
1. [Mansi Goel, Phd Scholar, IIIT Delhi](https://www.linkedin.com/in/mansigoel21/)
2. [Adarsh Singh Kushwah, M.Tech CSE, IIIT Delhi](https://www.linkedin.com/in/adarsh-singh-kushwah-59b119217/)
3. [Parul Sikri, M.Tech CSE, IIIT Delhi](https://www.linkedin.com/in/parul-sikri-8a45a3168/)
4. [Niharika, M.Tech CSE, IIIT Delhi](https://www.linkedin.com/in/niharika-201364168/)
5. [Shrey Rastogi, M.Tech CSE, IIIT Delhi](https://www.linkedin.com/in/shrey-rastogi-166b8a136/)

## ⚡Instructor
   [ Dr. Ganesh Bagler](https://en.wikipedia.org/wiki/Ganesh_Bagler)
<div align="center">
  <h3 align="center">Connect with Developers<img align="center" src="https://github.com/rajput2107/rajput2107/blob/master/Assets/Handshake.gif" height="33px" /></h3> 
</div>
<p align="center">
 <a href="https://www.linkedin.com/in/adarsh-singh-kushwah-59b119217/" target="blank">
  <img align="center" alt="LinkedIn" width="30px" src="https://github.com/hargun79/hargun79/blob/master/Assets/Linkedin.svg" /> &nbsp; &nbsp;
 </a>
 <a href="https://www.instagram.com/the_roy_heir/" target="blank">
  <img align="center" alt="insta" width="30px" src="https://github.com/SatYu26/SatYu26/blob/master/Assets/Instagram.svg" /> &nbsp; &nbsp;
 </a>
 <a href="https://www.facebook.com/shrey.rastogi.31" target="blank">
  <img align="center" alt="fb" width="35px" src="https://img.icons8.com/fluency/48/000000/meta.png" /> &nbsp; &nbsp;
 </a>
 <a href="https://github.com/mansigoel21" target="blank">
  <img align="center" alt="Github" width="37px" src="./Files Used in Readme/10415058851582692245.svg" />&nbsp; &nbsp;
 </a>
 <a href="https://api.whatsapp.com/send?phone=918130782929&text=Hi!" target="blank">
  <img align="center" alt="whatsappConnect" width="32px" src="./Files Used in Readme/Whatsapp-icon.png" /> &nbsp; &nbsp;
 </a>
 
  <br/>
  <br/>
  <br/>
</p>
<br/>
<p>
