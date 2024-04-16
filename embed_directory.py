import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


class VectorizationEngine:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str, pinecone_namespace: str):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        self.pinecone_namespace = pinecone_namespace

        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.embeddings_service = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    def wait_for_index(self):
        while not self.pinecone_client.describe_index(self.pinecone_index).status['ready']:
            time.sleep(1)
        return self.pinecone_client.Index(self.pinecone_index)

    def process_documents(self):
        # if not os.path.isdir('./train/faq.csv'):
        #     raise FileNotFoundError("Directory not found at specified document path.")

        loader = CSVLoader(file_path='./train/more.csv', encoding='utf8')
        documents = loader.load()

        print(f"Document Length: {len(documents)}")
        print("Embedding in progress...")

        PineconeVectorStore.from_documents(
            documents,
            self.embeddings_service,
            index_name=self.pinecone_index,
            namespace=self.pinecone_namespace,
        )
        print("Embedding finished.")


def load_environment():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", "PINECONE_API_KEY"),
        "pinecone_index": os.getenv("PINECONE_INDEX", "PINECONE_INDEX"),
        "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", "Mastery_Namespace"),
    }


def main():
    try:
        config = load_environment()

        print("Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        engine = VectorizationEngine(**config)
        engine.wait_for_index()
        engine.process_documents()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Unexpected error during embedding: {e}")


if __name__ == "__main__":
    main()