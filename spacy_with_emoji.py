import spacy
from spacy.tokens import Token
import emoji

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Register a new token extension to store emoji descriptions
Token.set_extension("is_emoji", default=False, force=True)

# Example usage
text = "I love pizza 🍕! The weather is 🌞."
doc = nlp(text)

def remove_duplicates_preserve_order(lemmas):
    seen = set()
    seen_add = seen.add
    return [x for x in lemmas if not (x in seen or seen_add(x))]

lemmas=[]
print(doc)
for token in doc:
        demojized_token = emoji.demojize(token.text)
        if demojized_token != token.text:
            lemmas.append(demojized_token)
        else:
            lemmas.append(token.lemma_)
lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
lemmas=remove_duplicates_preserve_order(lemmas)
print(" ".join(lemmas))