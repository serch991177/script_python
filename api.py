# api.py
import requests
import pandas as pd
import re
import unicodedata
import nltk
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from collections import OrderedDict
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar recursos de nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk_stopwords = set(stopwords.words('spanish'))

# Función de preprocesamiento
def apply_preprocessing(text, config):
    try:
        if config.get("lowercase", False):
            text = text.lower()
        if config.get("remove_accentuation", False):
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        if config.get("remove_punctuation", False):
            text = re.sub(r'[^\w\s]', '', text)
        if config.get("remove_numbers", False):
            text = re.sub(r'\d+', '', text)
        if config.get("remove_stopwords", False):
            text = " ".join(word for word in text.split() if word not in nltk_stopwords)
        if config.get("stemming", False):
            stemmer = SnowballStemmer("spanish")
            text = " ".join(stemmer.stem(word) for word in text.split())
        if config.get("remove_emojis", False):
            text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    except Exception as e:
        logging.error(f"Error en el preprocesamiento del texto: {e}")
        return text

# Función para preprocesar un DataFrame
def preprocessing(dataframe, fieldName, config):
    return dataframe[fieldName].apply(lambda text: apply_preprocessing(text, config))

# Función para transformar los datos
def transformData(data, fieldName, tokenizer, weight="TFIDF"):
    try:
        vectorizer_dict = {
            "TP": CountVectorizer(tokenizer=tokenizer, binary=True),
            "TF": CountVectorizer(tokenizer=tokenizer),
            "TFIDF": TfidfVectorizer(tokenizer=tokenizer)
        }
        vectorizer = vectorizer_dict.get(weight, TfidfVectorizer(tokenizer=tokenizer))
        X = vectorizer.fit_transform(data[fieldName])
        return vectorizer, X
    except Exception as e:
        logging.error(f"Error en la transformación de datos: {e}")
        return None, None

# Resto del código...

# Ruta para el análisis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        comments = data.get('comments', [])
        
        comments_df = pd.DataFrame({'comments': comments})
        comments_pp = pd.DataFrame({'comments': preprocessing(comments_df, fieldName='comments', config=config)})
        
        # Transformar los datos para modelado de temas
        vectorizer, X = transformData(comments_pp, fieldName="comments", tokenizer=word_tokenize, weight="TFIDF")
        if X is not None:
            feature_names = vectorizer.get_feature_names_out()
            text_collection = OrderedDict([(index, text) for index, text in enumerate(comments_pp["comments"])])
            corpus_index = [n for n in text_collection]
            bow = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)

            # Realizar el modelado de temas (LDA)
            LDA(bow, 5, 10)

            # Análisis de sentimiento utilizando un modelo BERT
            sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
            dfSenti = analyze_sentiment(comments_pp.comments.tolist(), sentiment_pipeline)

            # Clasificación de sentimientos
            dataLabel = dfSenti.polarity.tolist()
            negative = sum(1 for p in dataLabel if p == -1)
            neutral = sum(1 for p in dataLabel if p == 0)
            positive = sum(1 for p in dataLabel if p == 1)

            return jsonify({
                "positive": positive,
                "negative": negative,
                "neutral": neutral
            })
        else:
            return jsonify({"error": "Error en la transformación de los datos"}), 500
    except Exception as e:
        logging.error(f"Error en el análisis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
