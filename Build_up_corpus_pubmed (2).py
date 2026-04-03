import json
import sys
import os
# Temporarily add MedRAG repo
sys.path.append(os.path.abspath("../MedRAG"))
from src.medrag import MedRAG
from tqdm import tqdm
import re

print("--------------------- Start building corpus! ---------------------")
model_name_ = "OpenAI/gpt-35-turbo-16k"
retriever_name_ = "MedCPT"
corpus_name_ = "PubMed"
# corpus_name_ = "Textbooks"
output_name_ = "MedMCQA_train_top1k+" + model_name_ + "+" + retriever_name_ + "+" + corpus_name_
cot = MedRAG(llm_name=model_name_, rag=True, retriever_name=retriever_name_, corpus_name=corpus_name_)
