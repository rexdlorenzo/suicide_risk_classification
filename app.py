import pickle
import os
from cleaning_functions import cleaning_functions as cf
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(layout="wide")

svm_model_file = 'svm.pickle'

if os.path.exists(svm_model_file):
    with open(svm_model_file, 'rb') as f:
        svm_tfidf, svm_top_n, svm_selected_features, svm_best_model = pickle.load(f)

class_names = ['with no suicide risk', 'with suicide risk']
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_mask):
        self.feature_mask = feature_mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.feature_mask]


def svm_predictor (text):
    for func in cf.cleaning_functions:
        text = func(text)
    text = cf.tokenize_and_lemmatize(text)
    
    c = make_pipeline(svm_tfidf, FeatureSelector(svm_top_n), FeatureSelector(svm_selected_features), svm_best_model)
    
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, c.predict_proba, labels = [0, 1], num_features = 10)
    return c.predict([text]), c.predict_proba([text]), exp

st.title("Suicide Risk Classification")

inp, res = st.columns(2)

with inp:
    with inp.form(key='my_form', clear_on_submit=True):
        text_input = st.text_input(label='Text to analyze')
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            res.write(f"Input: {text_input}")
            prediction = svm_predictor(text_input)
            label = prediction[0][0]
            predict_proba = prediction[1][0][label]
            exp = prediction[2]
            icons = ['✅','🚨' ]
            res.info(f"Prediction: {class_names[label]}{icons[label]}")
            res.info(f"Prediction probability: {predict_proba:0.2%}")
            res.write("Top 10 features contributing to the prediction:")       
            plot = exp.as_pyplot_figure(label)
            res.pyplot(plot)
            
            res.markdown("*Features with positive values(green bars) contribute to the prediction being positive (with suicide risk)*")
            res.markdown("*Features with negative values(red bars) contribute to the prediction being negative (with no suicide risk)*")