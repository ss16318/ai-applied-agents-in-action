import os
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials


import tempfile
import chromadb


from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")


def process_documents():
    txt_files = []
    try:
        collection_names = chroma_client.list_collections()
        if collection_names:
            for name in collection_names:
                chroma_client.delete_collection(name=name)
            print(f"Successfully deleted {len(collection_names)} collections")
        else:
            print("No existing collections found.")

        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        credentials = Credentials(
            url=url,
            api_key=apikey,
        )
        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))  # Points to .../api/scripts
        docs_dir = os.path.abspath(os.path.join(current_dir, "..", "docs"))  # Points to .../api/docs

        for filename in os.listdir(docs_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(docs_dir, filename)  # ✅ Correct path
                with open(file_path, "rb") as f:
                    content = f.read()
                    txt_files.append(("files", (filename, content, "text/plain")))

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_tuple in txt_files:
                _, (filename, content, _) = file_tuple
                file_path = os.path.join(temp_dir, filename)
                file_name = os.path.splitext(filename)[0]

                try:
                    decoded = content.decode("utf-8")
                except UnicodeDecodeError as e:
                    print(f"❌ Could not decode {filename} as UTF-8: {e}")
                    continue

                with open(file_path, "w", encoding="utf-8") as buffer:
                    buffer.write(decoded)

                print(f"✅ Written decoded content to temp file: {file_path}")

                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()
                except Exception as e:
                    print(f"❌ Failed to load {file_path} with TextLoader: {e}")
                    continue

                print(f"📄 Loaded {len(documents)} documents from {filename}")


                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " "],
                    is_separator_regex=False,
                )

                
                splits = text_splitter.split_documents(documents)
                collection = chroma_client.get_or_create_collection(
                    name=file_name.lower(),
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 400,
                        "hnsw:M": 128,
                    },
                )

                batch_size = 100
                for i in range(0, len(splits), batch_size):
                    batch = splits[i : i + batch_size]
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    embeddings = embedding_model.embed_documents(
                        texts=texts, concurrency_limit=5
                    )
                    collection.add(
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                        ids=[f"{file_name}_{i}_{j}" for j in range(len(batch))],
                    )

        print(
            f"Successfully processed {len(txt_files)} files into their respective collections"
        )

        collection_names = chroma_client.list_collections()

        collections_info = []
        for name in collection_names:
            collection = chroma_client.get_collection(name)
            collections_info.append({"name": name, "count": collection.count()})

        print("Collections Info:", collections_info)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


def main():
    process_documents()


if __name__ == "__main__":
    main()
