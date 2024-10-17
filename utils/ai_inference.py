import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from utils.chroma_db import initialise_persistent_chromadb_client_and_collection, query_chromadb_collection

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def retry_on_rate_limit_error(max_retries=5, initial_delay=1, backoff_factor=2):
    def decorator_retry(func):
        def wrapper_retry(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
        return wrapper_retry
    return decorator_retry

@retry_on_rate_limit_error()
def gpt4o_mini_inference(system_prompt, instruction_prompt):

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ]
    )

    inference = completion.choices[0].message.content

    return inference

@retry_on_rate_limit_error()
def gpt4o_inference(system_prompt, instruction_prompt):

    completion = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ]
    )

    inference = completion.choices[0].message.content

    return inference

@retry_on_rate_limit_error()
def gpt4o_inference_with_search(system_prompt, instruction_prompt):

    tools_available = {
        "query_chromadb_collection": {
            "function": query_chromadb_collection,
            "description": "Query a vector database collection to retrieve relevant documents."
        }
    }

    tool = {
        "type": "function",
        "function": {
            "name": "query_chromadb_collection",
            "description": "Query a vector database collection to retrieve relevant documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "The collection to query. The value for this is always 'dd_documents'."
                    }, 
                    "query": {
                        "type": "string",
                        "description": "The query to the collection specifying what type of information is to be retrieved. It should be optimised for vector database lookup."
                    }, 
                    "n_results": {
                        "type": "number",
                        "description": "The number of results to retrieve. This must be a number between 10 and 20."
                    }
                }
            }
        }
    }

    tool_completion = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ],
        tools=[tool]
    )

    tool_call = tool_completion.choices[0].message.tool_calls[0]

    tool_name = tool_call.function.name

    tool_to_use = tools_available.get(tool_name).get("function")

    tool_description = tools_available.get(tool_name).get("description")

    arguments = json.loads(tool_call.function.arguments)

    print(f"\n\nTOOL ARGUMENTS: {arguments}\n\n")

    tool_answer = tool_to_use(
        initialise_persistent_chromadb_client_and_collection(arguments.get("collection")),
        arguments.get("query"),
        arguments.get("n_results")
    )

    completion = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""You received the following instruction:
            <instruction>
            {instruction_prompt}
            </instruction>
            You searched for and found the following documents:
            <documents>
            {tool_answer}
            </documents>
            Summarise the documents as they relate to the instruction. Irrelevant documents can be ignored. Relevant documents should be substantively reproduced to the extent that they provide valuable information with respect to the instruction.
            <references>
            You must include a reference to the source document in your summary for each document referred to.
            </references>
            """}
        ]
    )

    inference = completion.choices[0].message.content

    return inference