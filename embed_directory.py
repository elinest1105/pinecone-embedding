import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_NAMESPACE = "lancer"

def main():
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pc = Pinecone(api_key=PINECONE_KEY)
    # pinecone.init(api_key=PINECONE_KEY, environment=PINECONE_ENV)
    index_name = PINECONE_INDEX
    namespace = PINECONE_NAMESPACE

    loader = CSVLoader(file_path='./train/faq.csv', encoding='utf8')

    docs = loader.load()
    print("here1", len(docs))

    PineconeVectorStore.from_documents(
        docs,
        embeddings_model,
        index_name=index_name,
        namespace=namespace,
    )

    print("Embedding successful!")

if __name__ == "__main__":
    main() 