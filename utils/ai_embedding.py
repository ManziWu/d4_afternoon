import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

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
def e3_small_embedding(text_chunk):

    completion = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_chunk,
        encoding_format="float"
    )

    embedding = completion.data[0].get("embedding")

    return embedding

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
