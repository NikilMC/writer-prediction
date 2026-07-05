import joblib
import numpy as np
import re

from transformers import BertTokenizer, BertModel
from gensim.models import Word2Vec, FastText

import torch

# =====================================
# CLEAN TEXT
# =====================================
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =====================================
# LOAD SAVED MODEL
# =====================================
saved = joblib.load(
    "saved_models/LinearSVM_TF-IDF.pkl"
)

label_encoder = joblib.load(
    "saved_models/label_encoder.pkl"
)

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_author(text):

    text = clean_text(text)

    # =================================
    # TF-IDF / BoW
    # =================================
    if "vectorizer" in saved:

        vec = saved["vectorizer"]

        model = saved["model"]

        X = vec.transform([text])

    # =================================
    # WORD2VEC
    # =================================
    elif "w2v" in saved:

        model = saved["model"]

        w2v = saved["w2v"]

        tokens = text.split()

        valid = [
            w2v.wv[w]
            for w in tokens
            if w in w2v.wv
        ]

        if len(valid) == 0:
            X = np.zeros((1, 100))

        else:
            X = np.array([
                np.mean(valid, axis=0)
            ])

    # =================================
    # FASTTEXT
    # =================================
    elif "fasttext" in saved:

        model = saved["model"]

        ft = saved["fasttext"]

        tokens = text.split()

        valid = [
            ft.wv[w]
            for w in tokens
        ]

        X = np.array([
            np.mean(valid, axis=0)
        ])

    # =================================
    # BERT
    # =================================
    else:

        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        bert_model = BertModel.from_pretrained(
            'bert-base-uncased'
        )

        model = saved["model"]

        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = bert_model(**inputs)

        X = outputs.last_hidden_state[:, 0, :].numpy()

    # =================================
    # PREDICT
    # =================================
    pred = model.predict(X)[0]

    author = label_encoder.inverse_transform(
        [pred]
    )[0]

    return author

# =====================================
# TEST
# =====================================
sample = """
The Penne Arrabbiata Pasta is a delicious pasta which is made from a rich and spicy tomato basil sauce. The addition of the garlic, along with black pepper, red chilli flakes and the basil leaves, brings out the fresh flavours of this Tomato Basil Spicy Penne Arrabbiata Pasta.

Serve Penne Arrabbiata Pasta Recipe along with Thai Green Papaya Salad Recipe and a glass of wine for your weekend night dinner. Shop Archana's Kitchen Multi Millet Penne Pasta today and try this recipe and make your meals wholesome

If you liked the Penne Arrabbiata Pasta Recipe, do try our other Pasta Recipes : 

Pasta In Tomato Onion Chutney Recipe
Desi Style Masala Pasta Recipe (Indian Style Pasta)
Burnt Garlic Pasta with Mushroom Recipe
Penne Pasta In Marinara Sauce Recipe

Ingredients
2-1/2 cups Multi Millet Penne Pasta
1 Onion , finely chopped
4 cloves Garlic , finely chopped
2 cups Homemade tomato puree
1 tablespoon Red Chilli flakes
1 teaspoon Mixed Herbs (Dried)
1 teaspoon Whole Black Pepper Corns , coarsely pounded
Salt , to taste
1 teaspoon Sugar
3 tablespoons Extra Virgin Olive Oil
Basil leaves , a few sprigs

Instructions for Penne Arrabbiata Pasta Recipe Using Multi Millet Penne Pasta
1
To begin making the  Penne Arrabbiata Pasta Recipe we first have to cook the Archana's Kitchen Multi Millet Penne Pasta.

2
In a large pot, add water and bring it to a boil. Once the water has boiled, add a teaspoon of salt.

3
Add the Archana's Kitchen Multi Millet Penne Pasta to the boiling water and let it cook in the boiling water until it is al dente ( just cooked). This process should take a good 7 to 9 minutes.

4
Once the pasta has cooked, drain the water and run the cooked pasta under running water to stop the cooking process.

5
Drizzle some oil on top of pasta and make sure to coat all of the pasta with the oil so that the pasta doesn't stick to each other.

To make the Arrabiatta Pasta Sauce
1
Place a pan on medium heat, add oil and garlic and onion and saute for a few seconds until the onions turn soft.

2
Once the onions have softened, add the tomato puree, chilli flakes, mixed herbs, pounded black peppercorns, salt and sugar.

3
Cook the arrabbiata sauce together for a few minutes till the the sauce comes together and you can smell the aromas in the air. 

4
Add the cooked Penne Pasta to the spicy arrabbiata sauce and keep mixing till the sauce thickens and pasta is well coated with the sauce.

5
Once done, turn off the heat, check the salt and spices and adjust according to taste. Transfer the Penne Arrabbiata Pasta to a serving bowl. Finally sprinkle the parmesan cheese on the top and serve hot.

6
Serve Penne Arrabbiata Pasta Recipe along with Thai Green Papaya Salad Recipe and a glass of wine for your weekend night dinner.
"""

prediction = predict_author(sample)

print("Predicted Author:", prediction)