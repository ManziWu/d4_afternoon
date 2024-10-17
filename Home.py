__import__("pysqlite3") 
import sys 
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from utils.layout import page_config
from utils.ai_inference import gpt4o_inference_with_search, gpt4o_inference
from utils.chroma_db import initialise_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection

page_config()

if "log" not in st.session_state:
    st.session_state.log = ""

if "query" not in st.session_state:
    st.session_state.query = None

if "report" not in st.session_state:
    st.session_state.report = None

if "collection" not in st.session_state: 

    st.session_state.collection = initialise_persistent_chromadb_client_and_collection("dd_documents")

if "number_updates" not in st.session_state:

    st.session_state.number_updates = 0

def summary_agent(brief, report):

    ## TO-DO

def search_agent(instruction):

    ## TO-DO

    search_results = gpt4o_inference_with_search(SYSTEM_PROMPT, INSTRUCTION)

    return search_results

def lawyer_agent(brief, report=""):

    if st.session_state.number_updates == 5:

        st.markdown("Report Finalised")

        ##

        return final_report

    ## TO-DO

    search_instruction = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION_1)

    st.markdown("Briefing Search Agent")

    st.session_state.log += f"""
    ## SEARCH INSTRUCTION
    {search_instruction}
    \n\n
    """

    new_documents = search_agent(search_instruction)

    st.markdown("Reviewing Documents")

    st.session_state.log += f"""
    ## SEARCH RESULTS
    {new_documents}
    \n\n
    """

    ## TO-DO

    response = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION_2)

    st.markdown("Drafting Report")

    st.session_state.log += f"""
    ## LAWYER RESPONSE
    {response}
    \n\n
    """

    if "STOP" in response.upper() and st.session_state.number_updates > 1:

        report += response

        st.markdown("Report Finalised")

        final_report = summary_agent(brief, report)

        return final_report
    
    else:

        report += response

        st.markdown("Updating Report")

        st.session_state.number_updates = st.session_state.number_updates + 1

        return lawyer_agent(brief, report)

if st.session_state.query == None or st.session_state.query == "":

    st.markdown("## Brief")
    st.session_state.query = st.text_area(
        label="query",
        label_visibility="collapsed"
    )

if st.button("Run Brief"):

    st.session_state.report = lawyer_agent(st.session_state.query)
    st.session_state.number_updates = 0

if st.session_state.report is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## REPORT")

        with st.container(border=True):

            st.markdown(st.session_state.report)
    
    with col2:

        st.markdown("## LOG")

        with st.container(border=True):

            st.markdown(st.session_state.log)

