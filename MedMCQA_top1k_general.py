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

# model_name_ = "OpenAI/gpt-4o"
model_name_ = "OpenAI/gpt-35-turbo-16k"
retriever_name_ = "MedCPT"
# corpus_name_ = "MedCorp"
corpus_name_ = "Textbooks"
output_name_ = "MedMCQA_train_top1k+" + model_name_ + "+" + retriever_name_ + "+" + corpus_name_

data_train = []
with open("../MedMCQA/train.json", "r") as file:
    for line in file:
        question = json.loads(line)
        data_train.append(question)

data_train_top1k = data_train[:3]

def get_answer(cot, q):
    try:
        answer, _, _ = cot.answer(question=q['question'], options={"A": q['opa'], "B": q['opb'], "C": q['opc'], "D": q['opd']})
    except: 
        return None
    try:
        answer = json.loads(answer)
        return answer['answer_choice']
    except:
        list_of_string = re.findall(r'"(.*?)"', answer)
        if len(list_of_string) != 0:
            return list_of_string[-1]
        else:
            print('Output is not in json format!')
            print(answer)
            return None


cot = MedRAG(llm_name=model_name_, rag=True, retriever_name=retriever_name_, corpus_name=corpus_name_)
correct, incorrect = 0, 0
unknown = 0
true_labels = []
predicted_labels = []
for question in tqdm(data_train_top1k):
    a = get_answer(cot, question)
    mapper = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    true_labels.append(question['cop'])
    try:
        ans = (mapper[a[0]] == question['cop']) # True for correct answer, False otherwise
        predicted_labels.append(mapper[a[0]])
        if ans:
            correct += 1
        else:
            incorrect += 1
    except:
        predicted_labels.append(0)
        unknown += 1
        print(question)
        print(a)


label_encoder = LabelEncoder()
label_encoder.fit(true_labels+predicted_labels)
true_encoded = label_encoder.transform(true_labels)
predicted_encoded = label_encoder.transform(predicted_labels)
macro_f1 = f1_score(true_encoded, predicted_encoded, average='macro')

output = f"{output_name_}. # of Correct Answer: {correct}, # of Incorrect Answer: {incorrect}, # of Not-identified Pattern: {unknown}. Macro-f1: {macro_f1}."
print(output)
file_path = output_name_ + ".txt"
file_path = file_path.replace("/", "-")
with open(file_path, "w") as file:
    file.write(output)
