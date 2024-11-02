import re
import pandas as pd
import numpy as np
from string import punctuation
import spacy
from spacy.lang.en import stop_words
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sklearn.cluster import KMeans


class WithNerDocumentProcessor:
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
        self.stop_words = stop_words.STOP_WORDS
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
    user_input = "how to predict user input"

    # Calcula similaridade
    analyzer.calculate_similarities(user_input)

    top_docs = analyzer.top_n_similar_documents('similarity_mean', 100)

    kmeans = KMeans(n_clusters=5, random_state=42)
    top_embeddings = top_docs['title_summary_vector'].apply(analyzer.string_to_vector).tolist()
    kmeans.fit(top_embeddings)

    top_docs['Cluster'] = kmeans.labels_

    for cluster_num in range(5):
        cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity_mean'], ascending=[False])
        print(f"\nCluster {cluster_num}:")

        # Mostrar os 3 primeiros documentos do cluster
        print(f"Documentos do Cluster {cluster_num}:")
        for doc in cluster_docs[['Title', 'Summary', 'Link']].head(5).values:
            print(f" - Título: {doc[0]}")
            print(f" --- Link: {doc[2]}")
            print(f" -----  Resumo: {doc[1][:]}")
