import torch
from tqdm import tqdm
import streamlit as st
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class smallBERTMCR:
    def __init__(self, naq_threshold: float = 2.0):
        self.__bert_model = "mrm8488/bert-small-finetuned-squadv2"
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__bert_model)
        self.__model = AutoModelForQuestionAnswering.from_pretrained(self.__bert_model)
        self.__naq_threshold = naq_threshold

    def __classify_question(self, start_scores, end_scores) -> bool:
        score = min(max(start_scores[0]), max(end_scores[0]))
        if score < self.__naq_threshold:
            return False
        else:
            return True

    def extract_answers(self, context: str, questions: List[str]) -> Dict[str, List]:
        res_dict, answers = dict(), list()
        res_dict['context'] = context
        for question in tqdm(questions, desc="Extracting answers from context..."):
            res = dict()
            res['question'] = question
            inputs = self.__tokenizer(question, context, add_special_tokens=True, return_tensors="pt", max_length=512,
                                      truncation=True)
            input_ids = inputs["input_ids"].numpy()[0]
            outputs = self.__model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            res['is_answerable'] = self.__classify_question(answer_start_scores, answer_end_scores)
            if res['is_answerable']:
                # Get the most likely beginning and ends of answer with the argmax of the score
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                answer = self.__tokenizer.convert_tokens_to_string(
                    self.__tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                    
                #remove special tokens
                answer = [word.replace("▁","") if word.startswith("▁") else word for word in answer]
                answer = " ".join(answer).replace("[CLS]","").replace("[SEP]","").replace(" ##","")
                res['extracted_answer'] = answer.capitalize()
            else:
                res['extracted_answer'] = ""
            answers.append(res)
        res_dict['answers'] = answers
        return res_dict

@st.cache(hash_funcs={smallBERTMCR: lambda _: None}, allow_output_mutation=True)
def load_model():
    return smallBERTMCR()

MCR = load_model()

st.title("Machine Comprehension Reading Model")
st.write("A model to get answers from questions given a context. Powered by a small model of BERT.")

with st.form(key='input_form'):
    context = st.text_area('Context', height=300, help="Enter context here...")
    question = st.text_input('Question', help='All questions must be answerable based on the context.')
    submit_button = st.form_submit_button(label='Get Answer')
    if submit_button:
        res = MCR.extract_answers(context=context, questions=[question])
        st.text(" \n")
        st.text(" \n")
        st.markdown('**Answer:**')
        if res['answers'][0]['extracted_answer']:
            st.write(res['answers'][0]['extracted_answer'])
        else:
            st.write('Could not answer question.')
            #    st.write(f"Answer: {res['answers'][0]['extracted_answer']}")
