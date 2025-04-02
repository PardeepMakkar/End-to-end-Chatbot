from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams




#Extract data from the PDFs
def load_pdf(pdf_path):
    loader = DirectoryLoader(pdf_path,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#Create text chunks from documents
def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# Create chunks from pdf 
def prepare_pdf(pdf_path):
    documents = load_pdf(pdf_path)
    text_chunks = text_split(documents)
    print("length of my chunk:", len(text_chunks))
    return text_chunks


#download embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#Creating an in-memory vector database
def create_vector_store():
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )
    return vector_store


# Function to insert data into vector store
def load_vector_store(vector_store, text_chunks):
    ids = [x for x in range(len(text_chunks))]
    vector_store.add_documents(documents=text_chunks, ids=ids)
    return vector_store


def build_qa_bot(vector_store):
    prompt_template="""
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    model = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.2})
    
    qa_bot = RetrievalQA.from_chain_type(
            llm=model, 
            chain_type="stuff", 
            retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True, 
            chain_type_kwargs=chain_type_kwargs
        )
    
    return qa_bot


# Stitching components together to build qa chat bot
vector_store = create_vector_store()
text_chunks = prepare_pdf("data/")
load_vector_store(vector_store, text_chunks)
qa_bot = build_qa_bot(vector_store)


print("Model loaded successfully")


def generate_response(user_input: str):
    result = qa_bot({"query": user_input})
    return result["result"]
