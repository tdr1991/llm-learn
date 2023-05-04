#coding:utf-8

import requests
import os

import pandas as pd
import torch
import numpy as np
from sentence_transformers.util import semantic_search

model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
hf_token = os.getenv("HF_TOKEN")
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


def main():
    texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

    output = query(texts)
    # embeddings = pd.DataFrame(output)
    # print(embeddings)
    vec_len = np.linalg.norm(output, ord=2, axis=1)
    print(vec_len)
    src_embeddings = torch.FloatTensor(output)
    question = ["How can Medicare help me?"]
    output = query(question)
    query_embeddings = torch.FloatTensor(output)
    hits = semantic_search(query_embeddings, src_embeddings, top_k=5)
    print(hits)
    print([texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])


    


if __name__ == "__main__":
    main()
