import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# First-time setup
nltk.download("punkt")
nltk.download("wordnet")

text = "The cars were running faster than the other car."

lemmatizer = WordNetLemmatizer()
words = word_tokenize(text)

lemmas = [lemmatizer.lemmatize(w) for w in words]

print("Words:", words)
print("Lemmas:", lemmas)
