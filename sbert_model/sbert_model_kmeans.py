import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np


def string_to_vector(string_vector):
    cleaned_string = re.sub(r'[\[\]\n]', '', string_vector)
    return np.fromstring(cleaned_string.strip(), sep=' ')


def search_and_cluster(user_input, df, model, num_clusters=3):
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

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    top_embeddings = top_docs['title_summary_vector'].apply(string_to_vector).tolist()
    kmeans.fit(top_embeddings)

    top_docs['Cluster'] = kmeans.labels_

    with pd.ExcelWriter('sbert_kmeans_clusters.xlsx') as writer:
        for cluster_num in range(num_clusters):
            cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity'], ascending=False)

            cluster_df = cluster_docs[['Title', 'Link', 'Category']].head(5)

            cluster_df.to_excel(writer, sheet_name=f'Cluster_{cluster_num + 1}', index=False)

    for cluster_num in range(num_clusters):
        cluster_docs = top_docs[top_docs['Cluster'] == cluster_num].sort_values(by=['similarity'], ascending=[False])
        print(f"\nCluster {cluster_num + 1}:")

        print(f"Documentos do Cluster {cluster_num + 1}:")
        for doc in cluster_docs[['Title', 'Summary', 'Link', 'Primary Category', 'Category']].head(5).values:
            print(f" - Título: {doc[0]}")
            print(f" ---- Link: {doc[2]}")
            print(f" ------ Resumo: {doc[1][:]}")
            print(f" --------- Categorias: {doc[3]} {doc[4]}\n")


# Exemplo de execução
if __name__ == "__main__":
    file_path = './resources/sbert_vectors.csv'
    df = pd.read_csv(file_path)

    model = SentenceTransformer('all-mpnet-base-v2')

    user_input = "How can the security of web applications be improved against cyber attacks?"

    search_and_cluster(user_input, df, model)
