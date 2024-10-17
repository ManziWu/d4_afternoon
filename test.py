from utils.chroma_db import initialise_persistent_chromadb_client_and_collection, query_chromadb_collection

collection = initialise_persistent_chromadb_client_and_collection("dd_documents")

print(query_chromadb_collection(collection, "Jane Wu", 10))