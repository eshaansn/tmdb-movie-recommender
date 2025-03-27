import pandas as pd
from omegaconf import OmegaConf
import pickle


def make_features(config):
    print("Making features...")

    if config.task == 'use_llm_embeddings':

        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain.schema import Document

        from omegaconf import OmegaConf
        config = OmegaConf.load("./params.yaml")


        data = pickle.load(open(config.data.pr_data_unstemmed_save_path, 'rb'))
        data = pd.DataFrame(data)

        data['tagged_description'] = (data['id'].astype(str) + ' ' + data['tags'])

        documents = [Document(page_content=text) for text in data['tagged_description']]

        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(documents)


    
    elif config.task == 'create_vectors':

        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

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
