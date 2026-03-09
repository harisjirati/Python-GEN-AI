from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import UnstructuredURLLoader
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# constant Variables
CHUNK_SIZE = 1000
#   CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "real_estate_data"
llm = None
vector_store = None

##----------------------INITIALIZE COMPONENTS-----------------------------

def initialize_components():
    global llm, vector_store
    if llm is None:
       llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_tokens=500,
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )

##----------------------PROCESS URLS-----------------------------   

# def process_urls(urls):

#     """
#     :param urls: List of URLs to process
#     :return: None
#     """
#     yield ("Processing URLs...")
#     initialize_components()
#     try:
#         vector_store.delete_collection()
#     except Exception:
#         pass
    
#     # We must properly initialize the collection again or it throws NotFoundError
#     vector_store._collection = vector_store._client.create_collection(
#         name=COLLECTION_NAME,
#         embedding_function=vector_store._embedding_function,
#     )
#     loader = UnstructuredURLLoader(
#         urls=urls,
#         headers={
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
#         }
#     )
#     data = loader.load()
#     yield("Data loaded successfully")
#     text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "],chunk_size=CHUNK_SIZE, chunk_overlap=0)
#     docs = text_splitter.split_documents(data)
#   #  vectorstore = Chroma.from_documents(
#   #      documents=docs,
#   #      embedding=ef,
#   #      collection_name=COLLECTION_NAME,
#   #      persist_directory=str(VECTORSTORE_DIR),
#   #  )
#     uuids = [str(uuid4()) for _ in range(len(docs))]
#     vector_store.add_documents(documents=docs, ids=uuids)
#     yield("Data processed successfully")

def process_urls(urls):
    """
    :param urls: List of URLs to process
    """

    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading URLs..."
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    data = loader.load()

    yield "Splitting documents..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0
    )

    docs = text_splitter.split_documents(data)

    yield "Creating embeddings..."

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    yield "Processing complete ✅"

def generate_ans(query):
    if not vector_store:
        initialize_components()
    if not vector_store:
        raise RuntimeError("vector database is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question":query})
    sources = result.get("sources", "").split("\n")
    return result['answer'],sources    

##----------------------MAIN CODE-----------------------------

if __name__ == "__main__":
    initialize_components()
    urls = [
    #  "https://en.wikipedia.org/wiki/Artificial_intelligence",
     #"https://en.wikipedia.org/wiki/Generative_artificial_intelligence"
     ] 
    process_urls(urls) 

    answer, sources = generate_ans("what is AI")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")  
   # result = vector_store.similarity_search("What is AI")
   # print(result)   

