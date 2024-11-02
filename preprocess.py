import spacy
import nltk
import gensim
from string import punctuation

nlp = spacy.load("en_core_web_lg")
stop_words = gensim.parsing.preprocessing.STOPWORDS
punctuations = list(punctuation)


def tokenize_words(text):
    return nltk.word_tokenize(text)


def lemmatize_words_spacy(text):
    text = nlp(str(text))
    return " ".join([token.lemma_ for token in text])


def preprocess_text(df, df_column):
    df['text'] = df_column.progress_map(lemmatize_words_spacy)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].progress_map(tokenize_words)


def remove_stopwords(text, stopwords=stop_words):
    return [word for word in text if word not in stopwords and word not in punctuations and word.isalpha()]


def stopwords_treatment(df, df_column, stopwords=stop_words):
    df['text_stopwords_removed'] = df_column.apply(lambda x: remove_stopwords(x, stopwords))
    df['text_stopwords_removed'] = df['text_stopwords_removed'].apply(', '.join)
    df['text_stopwords_removed'] = df['text_stopwords_removed'].apply(
        lambda x: ' '.join(word for word in x.split() if len(word) > 3))


