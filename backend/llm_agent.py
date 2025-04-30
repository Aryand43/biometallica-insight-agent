import logging
import tiktoken
import re
import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os
from huggingface_hub import login
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN is not set in environment variables.")

login(token)
MODEL_NAME = "microsoft/phi-2" 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_PROMPT = """
You are an AI strategic analyst for a cleantech venture using synthetic biology and electrochemical engineering.

Based on the startup’s proposal below, generate a strategic insight report with these sections:

1. Executive Summary
2. SWOT Analysis
3. IP and Scientific Innovation Strategy
4. Commercialization & Go-to-Market Outlook
5. Team Readiness Assessment

Startup Proposal:
---
{content}
---
""".strip()


@st.cache_resource(show_spinner="Loading LLM model...")
def load_model_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def chunk_text_by_tokens(text, max_tokens=1000, buffer_tokens=50):
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        token_count = len(enc.encode(sentence))
        if current_tokens + token_count < max_tokens - buffer_tokens:
            current_chunk += sentence + " "
            current_tokens += token_count
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def clean_generated_output(generated_text, prompt):
    cleaned = generated_text.replace(prompt, "").strip()
    cleaned = re.sub(r'^.*?---\s*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def safe_generate(pipe, prompt, retries=2):
    for attempt in range(retries):
        try:
            return pipe(prompt, max_new_tokens=512, temperature=0.7, do_sample=True)[0]["generated_text"]
        except Exception as e:
            logging.warning(f"Retry {attempt+1}/{retries} failed: {e}")
            time.sleep(1)
    raise RuntimeError("LLM inference failed after retries.")


def generate_insight_report(text: str, max_chunk_tokens=1000):
    pipe = load_model_pipeline()
    tokenizer = pipe.tokenizer
    chunks = chunk_text_by_tokens(text, max_tokens=max_chunk_tokens)
    chunks = chunks[:5]  

    full_report = ""

    for i, chunk in enumerate(chunks):
        prompt = BASE_PROMPT.format(content=chunk)
        prompt_tokens = len(tokenizer.encode(prompt))
        logging.info(f"Generating Chunk {i+1}/{len(chunks)}")
        logging.info(f"Prompt length (chars): {len(prompt)} | Tokens: {prompt_tokens}")

        start = time.time()
        result = safe_generate(pipe, prompt)
        elapsed = time.time() - start
        logging.info(f"Finished Chunk {i+1} in {elapsed:.2f} seconds")

        cleaned = clean_generated_output(result, prompt)
        full_report += f"\n\n[Chunk {i+1} Insight]\n{cleaned}"

    return full_report.strip()
