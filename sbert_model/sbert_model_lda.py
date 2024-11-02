import re
import pandas as pd
import numpy as np

from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import gensim
from gensim import corpora
from gensim.models import LdaModel

from preprocess import  stopwords_treatment, preprocess_text

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


def get_doc_topic(text, lda, dictionary):
    doc_bow = dictionary.doc2bow(text)
    topics = lda.get_document_topics(doc_bow, minimum_probability=0.30)
    sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)[:3]
    return sorted_topics[0][0]


def search_and_cluster(user_input, df, model, num_clusters=5, stopwords=stopwords):
    # df['text'] = df['Title'] + ". " + df['Summary']
    #
    # document_embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    #
    # df['title_summary_vector'] = list(document_embeddings)
    #
    # df.to_csv('sbert_vectors.csv', index=False)

    user_embedding = model.encode([user_input])
    document_embeddings = df['title_summary_vector'].apply(string_to_vector).tolist()

    similarities = cosine_similarity(user_embedding, document_embeddings)[0]

    df['similarity'] = similarities

    top_docs = df.nlargest(100, 'similarity')

    preprocess_text(top_docs, top_docs['text'])

    stopwords_treatment(top_docs, top_docs['text'], stopwords)

    cnt = create_custom_stopwords(top_docs['text_stopwords_removed'])

    custom_stopwords = list(cnt.head(10).index)
    custom_stopwords = set(stopwords).union(set(custom_stopwords))

    stopwords_treatment(top_docs,  top_docs['text'], custom_stopwords)

    corpus, dictionary = lda_preproc(top_docs['text_stopwords_removed'])
    lda = lda_results(corpus, dictionary, num_topics=num_clusters)

    topic_df = get_topic_keywords(lda, num_words=8)

    top_docs['Cluster'] = [
        get_doc_topic(text.split(','), lda, dictionary)
        for text in top_docs['text_stopwords_removed'].tolist()
    ]

    for cluster_num in range(num_clusters):
        cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity'], ascending=[False])

        print(f"\nPalavras chaves do Cluster {cluster_num + 1}: " + ", ".join(topic_df[f"Cluster-{cluster_num}"].tolist()))

        print(f"Documentos do Cluster {cluster_num + 1}:")
        for doc in cluster_docs[['Title', 'Summary', 'Link']].head(5).values:
            print(f" - Título: {doc[0]}")
            print(f" ---- Link: {doc[2]}")
            print(f" ------  Resumo: {doc[1][:]}")


# Exemplo de execução
if __name__ == "__main__":
    file_path = './resources/sbert_vectors.csv'
    df = pd.read_csv(file_path)

    model = SentenceTransformer('all-mpnet-base-v2', )

    user_input = "nlp and word embedding techniques to search documents"

    search_and_cluster(user_input, df, model)
