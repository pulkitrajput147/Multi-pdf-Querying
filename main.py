'''
      End to End RAG LLM App using Llama Index and openai Indexing
'''

# Importing Necessary libraries
import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex,SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.response.pprint_utils import pprint_response

# Loading the environment variable
load_dotenv()

# Initializing API key
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Reading all the documents from the directory
documents=SimpleDirectoryReader("data").load_data()

# Converting Documents data  into Index
index=VectorStoreIndex.from_documents(documents)

# Creating a Query Engine
query_engine=index.as_query_engine()

response=query_engine.query("Your Query")
#pprint_response(response,show_source=True)       # for getting multiple results

retriever=VectorIndexRetriever(index=index,similarity_top_k=4)    # getting 4 different results
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.70)     # Setting certain threeshold
query_engine=RetrieverQueryEngine(retriever=retriever,node_postprocessors=[postprocessor])

response=query_engine.query("Your Query.")
pprint_response(response,show_source=True)


# storing the indexes in the hard disk
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query(" your query")
print(response)