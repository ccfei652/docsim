import pandas as pd
import ast
import re
import random

from preprocess import remove_stopwords, lemmatize_words_spacy

#for sentiment analysis
import spacy
import nltk

#for word clouds
from wordcloud import WordCloud, STOPWORDS

#LDA modeling
import gensim
from gensim import corpora
from gensim.models import LdaModel

#Visualisations
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models

#Settings, formatting, stopwords, and Spacy model
from tqdm.auto import tqdm
tqdm.pandas()
sns.set_palette("Accent")
pd.options.display.float_format = "{:,.3f}".format

stopwords = gensim.parsing.preprocessing.STOPWORDS

def string_to_vector(string_vector):
    return ast.literal_eval(string_vector)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(1, 25)


def show_cloud(df_column, max_words=200, stopwords=STOPWORDS, title=None, suptitle=None, save_name=None, show=False):
    words = " ".join(df_column.tolist())
    wordcloud = WordCloud(
        width=1200,
        height=800,
        max_words=max_words,
        background_color='white',
        stopwords=stopwords,
        random_state=42,
        collocation_threshold=4,
        min_word_length=2,
        min_font_size=16,
    ).generate(words)
    plt.figure(figsize=(16, 8), facecolor=None)
    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
    plt.axis("off")
    if save_name is not None:
        plt.savefig(save_name)
    if show is True:
        plt.show()


def lda_preproc(df_column):
    words = " ".join(df_column.tolist())
    words = words.split(", ")
    words = [t.split(" ") for t in words]
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=0.05, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in words]
    return corpus, dictionary


def lda_plot_results(corpus, id2word=None, num_topics=10, save_name=None, iterations=50, alpha='asymmetric'):
    lda = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=num_topics,
                   random_state=42,
                   iterations=iterations,
                   passes=5,
                   alpha=alpha,
                   per_word_topics=False)
    lda_display = pyLDAvis.gensim_models.prepare(lda, corpus, id2word, sort_topics=False)
    if save_name is not None:
        pyLDAvis.save_html(lda_display, save_name+'.html')
    pyLDAvis.display(lda_display)
    return pyLDAvis.display(lda_display), lda


def get_doc_topics(doc_bow, lda):
    topics = lda.get_document_topics(doc_bow, minimum_probability=0.40)
    sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)[:3]
    return [topic[0] for topic in sorted_topics]


def get_topic_keywords(lda_model, num_words=8):
    topic_words = {}
    for topic_id in range(lda_model.num_topics):
        topic_words[topic_id] = lda_model.show_topic(topic_id, num_words)

    topic_df = pd.DataFrame({i: [word[0] for word in topic_words[i]] for i in range(lda_model.num_topics)})
    topic_df.columns = [f'Cluster-{i}' for i in range(lda_model.num_topics)]

    return topic_words, topic_df


df = pd.read_csv('./resources/lda_example.csv')

# df['title_summary'] = df['Title'] + ". " + df['Summary']

df_lda = df.head(100).copy()

df_lda['text'] = df_lda['title_summary']
df_lda['text'] = df_lda['text'].progress_map(lemmatize_words_spacy)
df_lda['text'] = df_lda['text'].str.lower()
# df_lda['text'] = df_lda['text'].progress_map(tokenize_words)

# df_lda['text_stopwords_removed'] = df_lda['text'].progress_map(remove_stopwords)
# df_lda['text_stopwords_removed'] = df_lda['text_stopwords_removed'].apply(', '.join)
# df_lda['text_stopwords_removed'] = df_lda['text_stopwords_removed'].apply(lambda x: ' '.join(word for word in x.split() if len(word)>3))
#
# df_lda.to_csv('./resources/lda_example.csv', index=False)

from collections import Counter

words = " ".join(df_lda['text_stopwords_removed'].tolist())
words = words.split(", ")
cnt = Counter(words)
cnt = pd.DataFrame(dict(cnt), index=[0]).transpose().sort_values([0], ascending=False)
cnt.columns = ['count']
#
# plt.figure(figsize=(8, 6))
# sns.lineplot(y=cnt['count'], x=range(1, len(cnt)+1))
# plt.xlim(0, len(cnt))
# plt.title('Words count')
# plt.show()

custom_stopwords = list(cnt.head(10).index)
stopwords = set(stopwords).union(set(custom_stopwords))


df_lda['text_stopwords_removed'] = df_lda['text'].apply(lambda x: remove_stopwords(string_to_vector(x), stopwords))
df_lda['text_stopwords_removed'] = df_lda['text_stopwords_removed'].apply(', '.join)
df_lda['text_stopwords_removed'] = df_lda['text_stopwords_removed'].apply(lambda x: ' '.join(word for word in x.split() if len(word)>3))

# show_cloud(df_lda['text_stopwords_removed'], max_words=200, show=True)

corpus, dictionary = lda_preproc(df_lda['text_stopwords_removed'])
vis, lda = lda_plot_results(corpus=corpus, id2word=dictionary, num_topics=5, save_name='LDA results')

topic_words, topic_df = get_topic_keywords(lda, num_words=8)

df_lda['Cluster'] = [get_doc_topics(dictionary.doc2bow(text.split(',')), lda) for text in df_lda['text_stopwords_removed'].tolist()]

print()