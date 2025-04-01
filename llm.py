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
from uuid import uuid4



#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents




#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings



def prepare_pdf():
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)
    print("length of my chunk:", len(text_chunks))
    return text_chunks


embeddings = download_hugging_face_embeddings()


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

def load_vector_store(vector_store, documents):
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store




def load_llm():
    prompt_template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs={"prompt": PROMPT}


    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})
    
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)
    return qa



vector_store = create_vector_store()
documents = prepare_pdf()
vector_store = load_vector_store(vector_store, documents=documents)
qa = load_llm()

print("Model loaded successfully")

def generate_response(user_input: str):
    result=qa({"query": user_input})
    return result["result"]


