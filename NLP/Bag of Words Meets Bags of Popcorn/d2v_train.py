# %%
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
# %%
def review_to_wordlist(raw_text, remove_stopwords = False):
    soup = BeautifulSoup(raw_text)
    without_html = soup.get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', without_html)
    words = letters_only.lower().split()

    if remove_stopwords is True:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words

# %%
if __name__ == "__main__":
    train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)

    test = pd.read_csv("data/testData.tsv", header=0,
                       delimiter="\t", quoting=3)

    unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0,
                                  delimiter="\t", quoting=3)

# %%
    tagged_all_data = []

    print("Очистка и тегирование размеченных (TRAIN) отзывов...")
    for i, review in enumerate(train["review"]):
        words = review_to_wordlist(review, remove_stopwords=True)
        tagged_all_data.append(TaggedDocument(words=words, tags=[f"TRAIN_{i}"]))

    print("Очистка и тегирование неразмеченных (UNLABELED) отзывов...")
    for i, review in enumerate(unlabeled_train["review"]):
        words = review_to_wordlist(review, remove_stopwords=True)
        tagged_all_data.append(TaggedDocument(words=words, tags=[f"UNLABELED_{i}"]))

# %%
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    print("Инициализация и сборка словаря Doc2Vec...")
    model = Doc2Vec(
        vector_size = 300,
        window = 10,
        min_count = 40,
        workers = 4,
        dm = 0,          # dm=0 означает режим PV-DBOW
        epochs = 10
    )

    model.build_vocab(tagged_all_data)

    print("Обучение Doc2Vec (это займет пару минут)...")
    model.train(
        tagged_all_data,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    model.save("models/300features_40minwords_10context_d2v")
    print("Doc2Vec успешно обучен и сохранен!")