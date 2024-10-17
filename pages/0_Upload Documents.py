__import__("pysqlite3") 
import sys 
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import zipfile
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from utils.layout import page_config
from utils.chroma_db import initialise_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection

page_config()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "vectordb_collection" not in st.session_state:
    st.session_state.vectordb_collection = initialise_persistent_chromadb_client_and_collection("dd_documents")

st.markdown("#### Upload Files")

st.session_state.uploaded_files = st.file_uploader(
    "Upload a ZIP File",
    type="zip"
)

if st.session_state.uploaded_files is not None:

    if st.button("Embed Documents and Load to Chroma"):

        with zipfile.ZipFile(st.session_state.uploaded_files, 'r') as zip_ref:
            
            zip_ref.extractall('extracted_files')

            num_docs = len(zip_ref.namelist())

            st.markdown(f"NUM DOCS: {num_docs}")

            doc_num = 1
            
            for document_name in zip_ref.namelist():

                st.markdown(f"HANDLING DOC #: {doc_num}")

                doc_num = doc_num + 1

                if document_name.startswith("__MACOSX"):

                    continue

                with zip_ref.open(document_name) as extracted_file:

                    document_content = extracted_file.read().decode(
                        encoding="utf-8", 
                        errors="replace"
                    )
                
                document = [
                    Document(
                        text=document_content
                    )
                ]

                splitter = TokenTextSplitter(
                    chunk_size=256,
                    chunk_overlap=52,
                    separator=" "
                )

                nodes = splitter.get_nodes_from_documents(document)

                for node in nodes:

                    add_document_chunk_to_chroma_collection(st.session_state.vectordb_collection, document_name, node.text, node.node_id)