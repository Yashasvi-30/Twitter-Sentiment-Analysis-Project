import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", '', text.lower())
    return ' '.join([stemmer.stem(w) for w in text.split() if w not in stop_words])
