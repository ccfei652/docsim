import re
import pandas as pd
import numpy as np
import ast

import spacy
from gensim.models import KeyedVectors

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim import corpora
from gensim.models import LdaModel

from preprocess import lemmatize_words_spacy, remove_stopwords, tokenize_words

from string import punctuation

from tqdm.auto import tqdm

tqdm.pandas()
pd.options.display.float_format = "{:,.3f}".format

stopwords = gensim.parsing.preprocessing.STOPWORDS


def string_to_vector(string_vector):
    cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)
    return np.fromstring(cleaned_string.strip(), sep=' ')


def create_custom_stopwords(df_column):
    words = " ".join(df_column.tolist())
    words = words.split(", ")
    cnt = Counter(words)
    cnt = pd.DataFrame(dict(cnt), index=[0]).transpose().sort_values([0], ascending=False)
    cnt.columns = ['count']

    return cnt


def lda_preproc(df_column):
    words = " ".join(df_column.tolist())
    words = words.split(", ")
    words = [t.split(" ") for t in words]
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    corpus = [dictionary.doc2bow(text) for text in words]
    return corpus, dictionary


def lda_results(corpus, id2word=None, num_topics=5, iterations=50, alpha='asymmetric'):
    lda = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=num_topics,
                   random_state=42,
                   iterations=iterations,
                   passes=5,
                   alpha=alpha,
                   per_word_topics=False)
    return lda


def get_topic_keywords(lda_model, num_words=8):
    topic_words = {}
    for topic_id in range(lda_model.num_topics):
        topic_words[topic_id] = lda_model.show_topic(topic_id, num_words)

    topic_df = pd.DataFrame({i: [word[0] for word in topic_words[i]] for i in range(lda_model.num_topics)})
    topic_df.columns = [f'Cluster-{i}' for i in range(lda_model.num_topics)]

    return topic_df


def get_doc_topic(doc_bow, lda):
    topics = lda.get_document_topics(doc_bow, minimum_probability=0.30)
    sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)[:3]
    return sorted_topics[0][0]


class WithNerDocumentProcessor:
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
        self.stop_words = stopwords
        self.punctuations = list(punctuation)

    def vectorize_text(self, text):
        words = self.preprocess(text)
        return self.document_vector(words)

    def preprocess(self, text):
        sentence = self.nlp(text)
        # lemmatization
        sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
        # removing stop words
        sentence = [
            word for word in sentence
            if word not in self.stop_words
               and word not in self.punctuations and word.isalpha()
        ]
        return sentence

    def document_vector(self, sentence: list[str]):
        result_vector = np.zeros(300)
        model_words = list(self.model.index_to_key)

        for word in sentence:
            if word in model_words:
                result_vector += self.model.get_vector(word)
                continue
            if word.lower() in model_words:
                result_vector += self.model.get_vector(word.lower())
                continue
            if word.title() in model_words:
                result_vector += self.model.get_vector(word.title())
                continue

        return result_vector

    @staticmethod
    def calculate_similarity(vector1, vector2):
        return cosine_similarity([vector1], [vector2]).flatten()[0]


class WithNerSimilarityAnalyzer:
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    @staticmethod
    def string_to_vector(string_vector):
        cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)
        return np.fromstring(cleaned_string.strip(), sep=' ')

    @staticmethod
    def string_to_object(string_object):
        return ast.literal_eval(string_object)

    def add_title_plus_summary_column(self):
        self.df['Title_Summary'] = self.df['Title'] + ". " + self.df['Summary']

    def extract_and_add_entities_column(self):
        self.df['entities'] = self.df['Title_Summary'].apply(lambda x: self.processor.extract_entities(x))

    def extract_and_add_linked_entities_column(self):
        self.df['linked_entities'] = self.df['Title_Summary'].apply(lambda x: self.processor.extract_linked_entities(x))

    def apply_vectorization(self):
        self.df['title_vectors'] = self.df['Title'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['summary_vector'] = self.df['Summary'].apply(lambda x: self.processor.vectorize_text(x))
        self.df['title_summary_vector'] = self.df['Title_Summary'].apply(lambda x: self.processor.vectorize_text(x))

    def calculate_similarities(self, user_input):
        user_input_vector = self.processor.vectorize_text(user_input)
        self.df['title_similarity'] = self.df['title_vectors'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, self.string_to_vector(x)))
        self.df['summary_similarity'] = self.df['summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, self.string_to_vector(x)))
        self.df['title_summary_similarity'] = self.df['title_summary_vector'].apply(
            lambda x: self.processor.calculate_similarity(user_input_vector, self.string_to_vector(x)))

        self.df['similarity_mean'] = self.df[
            ['title_similarity', 'summary_similarity', 'title_summary_similarity']].mean(axis=1)

    def top_n_similar_documents(self, similarity_column, n=10):
        return self.df.sort_values(by=[similarity_column], ascending=[False]).head(n)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")

    word2vec_model = KeyedVectors.load('./resources/word2vec-google-news-300.model')

    df = pd.read_csv('./resources/titles_vectors.csv')

    # Inicializar processador e analisador
    processor = WithNerDocumentProcessor(model=word2vec_model, nlp=nlp)
    analyzer = WithNerSimilarityAnalyzer(df=df, processor=processor)

    # Adicionar a coluna de entidades
    # analyzer.add_title_plus_summary_column()

    # Aplicar vetorização
    # analyzer.apply_vectorization()

    # Salvar no CSV
    # df.to_csv('titles_vectors.csv', index=False)

    # Definir input do usuário
    user_input = "Techniques for learning and interpreting natural language and generating automatic responses in artificial intelligence systems"

    # Calcula similaridade
    analyzer.calculate_similarities(user_input)

    top_docs = analyzer.top_n_similar_documents('similarity_mean', 100)

    top_docs['text'] = top_docs['Title_Summary'].progress_map(lemmatize_words_spacy)
    top_docs['text'] = top_docs['text'].str.lower()
    top_docs['text'] = top_docs['text'].progress_map(tokenize_words)

    top_docs['text_stopwords_removed'] = top_docs['text'].progress_map(remove_stopwords)
    top_docs['text_stopwords_removed'] = top_docs['text_stopwords_removed'].apply(', '.join)
    top_docs['text_stopwords_removed'] = top_docs['text_stopwords_removed'].apply(
        lambda x: ' '.join(word for word in x.split() if len(word) > 3))

    cnt = create_custom_stopwords(top_docs['text_stopwords_removed'])

    custom_stopwords = list(cnt.head(10).index)
    custom_stopwords = set(stopwords).union(set(custom_stopwords))

    top_docs['text_stopwords_removed'] = top_docs['text'].apply(
        lambda x: remove_stopwords(x, custom_stopwords)
    )
    top_docs['text_stopwords_removed'] = top_docs['text_stopwords_removed'].apply(', '.join)
    top_docs['text_stopwords_removed'] = top_docs['text_stopwords_removed'].apply(
        lambda x: ' '.join(word for word in x.split() if len(word) > 3))

    corpus, dictionary = lda_preproc(top_docs['text_stopwords_removed'])
    lda = lda_results(corpus, dictionary, num_topics=3)

    topic_df = get_topic_keywords(lda, num_words=8)

    top_docs['Cluster'] = [
        get_doc_topic(dictionary.doc2bow(text.split(',')), lda) for text in top_docs['text_stopwords_removed'].tolist()
    ]

    with pd.ExcelWriter('google_lda_clusters.xlsx') as writer:
        for cluster_num in range(3):
            cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity_mean'], ascending=False)

            cluster_df = cluster_docs[['Title', 'Summary', 'Link', 'Primary Category', 'Category']].head(5)

            cluster_df.to_excel(writer, sheet_name=f'Cluster_{cluster_num + 1}', index=False)

    for cluster_num in range(3):
        cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity_mean'], ascending=[False])

        print(f"\nPalavras chaves do Cluster {cluster_num + 1}: " + ", ".join(
            topic_df[f"Cluster-{cluster_num}"].tolist()))

        print(f"Documentos do Cluster {cluster_num + 1}:")
        for doc in cluster_docs[['Title', 'Summary', 'Link', 'Primary Category', 'Category']].head(5).values:
            print(f" - Título: {doc[0]}")
            print(f" ---- Link: {doc[2]}")
            print(f" ------ Resumo: {doc[1][:]}")
            print(f" --------- Categorias: {doc[3]} {doc[4]}")
            print("\n")

