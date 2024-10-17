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
    SYSTEM_PROMPT = "You are a legal expert tasked with summarizing a mergers and acquisitions transaction report. Summarize the key risks and findings from the following brief and report."
    
    # 生成指令，汇总 brief 和 report
    INSTRUCTION = f"Summarize the following brief and report:\n\nBrief:\n{brief}\n\nReport:\n{report}"

    # 调用 GPT 模型生成总结
    summary_response = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION)

    return summary_response
    

def search_agent(instruction):

    ## TO-DO
# 构造系统提示，让 GPT 模型明白这是一个搜索代理任务
    SYSTEM_PROMPT = "You are a search agent tasked with finding the most relevant documents based on the instruction provided."
    
    # 将指令传递给 GPT 模型以生成搜索内容
    INSTRUCTION = instruction

    # 指定希望返回的搜索结果数
    n_results = 5  # 你可以根据需要调整返回的结果数

    # 使用 ChromaDB 的查询功能，根据指令查找最相关的文档块
    search_results = query_chromadb_collection(st.session_state.collection, INSTRUCTION, n_results=n_results)

    # 打印或显示搜索的日志信息，便于调试
    st.session_state.log += f"""
    ## SEARCH RESULTS
    {search_results}
    \n\n
    """

    # 返回查询结果
    return search_results

def lawyer_agent(brief, report=""):

    if st.session_state.number_updates == 5:

        st.markdown("Report Finalised")

        ##
        final_report = summary_agent(brief, report)
        return final_report


    ## TO-DO
    SYSTEM_PROMPT = "You are a lawyer advising a client on an M&A transaction. Generate a search instruction based on the brief."
    INSTRUCTION_1 = f"Generate a search instruction for the following brief: {brief}"


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
    # 使用 search_agent 通过搜索指令检索相关文档
    new_documents = search_agent(search_instruction)

    st.markdown("Reviewing Documents")

    # 记录搜索结果
    st.session_state.log += f"""
    ## SEARCH RESULTS
    {new_documents}
    \n\n
    """

    # 初始化 response 变量以存储报告部分内容
    response = ""

    # 生成报告的下一步
    SYSTEM_PROMPT = "You are a lawyer drafting a report based on the client's brief and the new documents found."
    INSTRUCTION_2 = f"Generate a report based on the brief and the following documents: {new_documents}"

    # 尝试调用 GPT 模型生成报告片段
    try:
        response = gpt4o_inference(SYSTEM_PROMPT, INSTRUCTION_2)
    except Exception as e:
        st.error(f"Error during GPT inference: {e}")

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

