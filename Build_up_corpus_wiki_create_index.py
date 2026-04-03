import json
import sys
import os
# Temporarily add MedRAG repo
sys.path.append(os.path.abspath("../MedRAG"))
from src.medrag import MedRAG
from tqdm import tqdm
import re
import faiss
import json
import torch
import tqdm
import numpy as np

print("--------------------- Start building corpus! ---------------------")

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32):

    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")
    
    if HNSW:
        M = M
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        if "specter" in model_name.lower():
            index = faiss.IndexFlatL2(h_dim)
        else:
            index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index

class Generate_embed: 

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwarg):
        self.corpus_name = "wikipedia"
        self.retriever_name = "ncbi/MedCPT-Query-Encoder"
        self.index_dir = "./corpus/wikipedia/index/ncbi/MedCPT-Article-Encoder"

        if self.corpus_name in ["textbooks", "pubmed", "wikipedia"] and self.retriever_name in ["allenai/specter", "facebook/contriever", "ncbi/MedCPT-Query-Encoder"] and not os.path.exists(os.path.join(self.index_dir, "embedding")):
            print("[In progress] Downloading the {:s} embeddings given by the {:s} model...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
            os.makedirs(self.index_dir, exist_ok=True)
            if self.corpus_name == "textbooks":
                if self.retriever_name == "allenai/specter":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EYRRpJbNDyBOmfzCOqfQzrsBwUX0_UT8-j_geDPcVXFnig?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "facebook/contriever":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQqzldVMCCVIpiFV4goC7qEBSkl8kj5lQHtNq8DvHJdAfw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
            elif self.corpus_name == "pubmed":
                if self.retriever_name == "allenai/specter":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ebz8ySXt815FotxC1KkDbuABNycudBCoirTWkKfl8SEswA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "facebook/contriever":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EWecRNfTxbRMnM0ByGMdiAsBJbGJOX_bpnUoyXY9Bj4_jQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
            elif self.corpus_name == "wikipedia":
                if self.retriever_name == "allenai/specter":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ed7zG3_ce-JOmGTbgof3IK0BdD40XcuZ7AGZRcV_5D2jkA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "facebook/contriever":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/ETKHGV9_KNBPmDM60MWjEdsBXR4P4c7zZk1HLLc0KVaTJw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                    os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
            os.system("unzip {:s} -d {:s}".format(os.path.join(self.index_dir, "embedding.zip"), self.index_dir))
            os.system("rm {:s}".format(os.path.join(self.index_dir, "embedding.zip")))
            h_dim = 768
        else:
            h_dim = 768
            print("No need to generate embedding! Proceed to index.")
            # h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwarg)
    
        print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
        self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim, HNSW=True)
        print("[Finished] Corpus indexing finished!")
        self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]            


Generate_embed()


"""
model_name_ = "OpenAI/gpt-35-turbo-16k"
retriever_name_ = "MedCPT"
corpus_name_ = "Wikipedia"
# corpus_name_ = "Textbooks"
output_name_ = "MedMCQA_train_top1k+" + model_name_ + "+" + retriever_name_ + "+" + corpus_name_
cot = MedRAG(llm_name=model_name_, rag=True, retriever_name=retriever_name_, corpus_name=corpus_name_)
"""