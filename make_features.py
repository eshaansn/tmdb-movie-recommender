import pandas as pd
from omegaconf import OmegaConf
import pickle
from dotenv import load_dotenv
import os



def make_features(config):
    print("Making features...")

    if config.features.method == 'use_llm_embeddings':

        print('Using LLM embeddings...')

        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain.schema import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma  import Chroma


        data = pickle.load(open(config.data.pr_data_unstemmed_save_path, 'rb'))
        data = pd.DataFrame(data)

        data['tagged_description'] = (data['id'].astype(str) + ' ' + data['tags'])

        documents = [Document(page_content=text) for text in data['tagged_description']]

        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(documents)
        
        persist_dir = config.features.documentstore 

        load_dotenv()

        if os.path.exists(persist_dir) and os.listdir(persist_dir): 
            print("Chroma database already exists.")
        else:
            print("Creating a new Chroma database...")
            embedding_model = HuggingFaceEmbeddings(model_name=config.features.model_name)
            ids = [str(x).split("=")[1].split()[0].strip("'") for x in documents]
            db_movies = Chroma.from_documents(
                documents,
                collection_name='TMDB5000',
                ids=ids,
                embedding=embedding_model,
                persist_directory=persist_dir,
                create_collection_if_not_exists=True,
    )


    
    elif config.features.method == 'create_vectors':
        print('Creating vectors...')

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

        pickle.dump(vectors, open(config.features.features_save_path, 'wb'))
        pickle.dump(similarity, open(config.features.similarity_save_path, 'wb'))

    

if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    make_features(config)
