import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, AutoModelForSequenceClassification
import torch
MODEL_PATH = './model'
DATA_PATH = './data/data.pkl'
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
DATA = pd.read_pickle(DATA_PATH)
ACTIONS = list(DATA['Action'].unique())
TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH)
PS= PorterStemmer()

def infer(text:str):
    validity = get_validity(text)
    if not validity:
        st.error('Commentary not valid', icon="🚨")
        return
    action = get_action(text)
    st.success(f'Commentary valid ! Action: **{action}**', icon="✅")

def get_validity(text:str) -> bool:
    """We perform validity inference by feeding forward our fine tuned mode.
    Args:
        text (str): A commentary

    Returns:
        bool: True if the commentary is valide False if he is not
    """
    inputs = TOKENIZER(text, return_token_type_ids=False, return_tensors='pt')
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    validity_class = np.argmax(list(logits))
    return bool(validity_class)

def get_action(text:str) -> str:
    """We perform action inference by apllying stemming to the commentary and then check if an action is present in the text.

    Args:
        text (str): A commentary

    Returns:
        str: Action detected
    """
    words = text.split(' ')
    stem_list = []
    for word in words:
        stem_list.append(PS.stem(word))
    stemmed_text = ' '.join(stem_list)
    for action in ACTIONS:
        if action in stemmed_text:
            return action
    return 'No action detected'

st.title('NLP team\'s task inference !')
st.write('    ')
st.write('    ')
st.write('    ')
st.write('    ')


input = st.text_input('Try a commentary', 'Wemby dunked over everyone !')
if st.button('Is it describing any action ?'):
    infer(input)

