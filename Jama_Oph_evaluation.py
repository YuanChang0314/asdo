import json
import sys
import os
# Temporarily add MedRAG repo
sys.path.append(os.path.abspath("../MedRAG"))
from src.medrag import MedRAG
from tqdm import tqdm
import re
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def standardize_question(text):
    pattern = r"Question: (.*?)\n(A\..*?)(\nB\..*?)(\nC\..*?)(\nD\..*)"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        question = match.group(1)
        choices = {
            "A": match.group(2).strip()[3:],  # Remove 'A. ' from the beginning
            "B": match.group(3).strip()[3:],  # Remove 'B. ' from the beginning
            "C": match.group(4).strip()[3:],  # Remove 'C. ' from the beginning
            "D": match.group(5).strip()[3:]   # Remove 'D. ' from the beginning
        }
        return question, choices
    else:
        print("[ERROR]: ", text)


def get_answer(cot, q, c):
    try:
        answer, _, _ = cot.answer(question=q, options={"A": c['A'], "B": c['B'], "C": c['C'], "D": c['D']})
    except: 
        return None
    try:
        answer = json.loads(answer)
        return answer['answer_choice']
    except:
        if isinstance(answer, str):
            list_of_string = re.findall(r'"(.*?)"', answer)
            if len(list_of_string) != 0:
                return list_of_string[-1]
            print('Output is not in json format or answer is not string!')
            print(answer)
            return None
        
            

model_names = ["OpenAI/gpt-35-turbo-16k", "OpenAI/gpt-4o"]
# model_names = ["OpenAI/gpt-4o"]
retriever_names = ["BM25", "MedCPT"]
# retriever_names = ["Contriever", "SPECTER"]
corpus_names = ["PubMed", "StatPearls", "Textbooks", "Wikipedia", "MedCorp"]
# corpus_names = ["MedCorp"]

with open('../data/results/jama_ophthalmology_clinical_challenge_378_result.json', 'r') as file:
    data = json.load(file)

# Test without RAG
for model_name_ in model_names:
    output_name_ = "JAMA(Ophthalmology)+" + model_name_ + "+CoT"
    print('---', output_name_, '---')
    cot = MedRAG(llm_name=model_name_, rag=False)
    for i, question in tqdm(enumerate(data)):
        if output_name_ in data[i].keys():
            continue
        else:
            q, c = standardize_question(question['query'])
            a = get_answer(cot, q, c)
            data[i][output_name_] = a
        with open('../data/results/jama_ophthalmology_clinical_challenge_378_result.json', 'w') as file:
                json.dump(data, file)

for model_name_ in model_names:
    for retriever_name_ in retriever_names:
        for corpus_name_ in corpus_names:
            output_name_ = "JAMA(Ophthalmology)+" + model_name_ + "+" + retriever_name_ + "+" + corpus_name_
            print('---', output_name_, '---')
            cot = MedRAG(llm_name=model_name_, rag=True, retriever_name=retriever_name_, corpus_name=corpus_name_)
            correct, incorrect = 0, 0
            unknown = 0
            true_labels = []
            predicted_labels = []
            for i, question in tqdm(enumerate(data)):
                if output_name_ in data[i].keys():
                    continue
                else:
                    q, c = standardize_question(question['query'])
                    a = get_answer(cot, q, c)
                    data[i][output_name_] = a

            with open('../data/results/jama_ophthalmology_clinical_challenge_378_result.json', 'w') as file:
                json.dump(data, file)