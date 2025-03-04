import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def make_features(config):
    print("Making features...")

    data = pickle.load(open(config.data.pr_data_save_path, 'rb'))
    data = pd.DataFrame(data)

    vectorizer_name = config.features.vectorizer
    vectorizer = {
        "count-vectorizer": CountVectorizer,
        "tfidf-vectorizer": TfidfVectorizer
    }[vectorizer_name](stop_words="english")

    vectors = vectorizer.fit_transform(data.tags).toarray()
    similarity = cosine_similarity(vectors)

    pickle.dump(vectors, open(config.data.features_save_path, 'wb'))
    pickle.dump(similarity, open(config.data.similarity_save_path, 'wb'))

    

if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    make_features(config)
