#%%
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
import logging

#%%
def review_to_wordlist(raw_text, remove_stopwords = False):
    soup = BeautifulSoup(raw_text)
    without_html = soup.get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', without_html)
    words = letters_only.lower().split()

    if remove_stopwords is True:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words

#%%
def review_to_sentences(review, tokenizer, remove_stopwords = False):
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences

# %%
if __name__ == "__main__":
    train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)

    unlabeled_train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                                  delimiter="\t", quoting=3)

    test = pd.read_csv("data/testData.tsv", header=0,
                       delimiter="\t", quoting=3)

    nltk.download()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# %%
    wordlist_reviews = []

    for review in train['review']:
        wordlist_reviews += review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        wordlist_reviews += review_to_sentences(review, tokenizer)

# %%
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    print("Инициализация и сборка словаря Word2Vec...")
    model = word2vec.Word2Vec(
        workers = 4,
        vector_size = 300,
        min_count = 40,
        window = 10,
        sample = 1e-3
    )

    model.build_vocab(wordlist_reviews)

    print("Обучение Word2Vec (это займет пару минут)...")
    model.train(
        wordlist_reviews,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    model.init_sims(replace=True)

    model.save("models/300features_40minwords_10context_w2v")
    print("Word2Vec успешно обучен и сохранен!")