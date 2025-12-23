import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import time
from datetime import datetime

st.set_page_config(page_title="Detector de Frau Bancari", page_icon="🔒")

@st.cache_resource
def load_artifacts():
    columns = joblib.load('model_columns.pkl')
    
    scaler = joblib.load('scaler.pkl')
    
    model = TabNetClassifier()

    model.load_model('./tabnet_model.zip')
    
    return columns, scaler, model


model_columns, scaler, model = load_artifacts()
st.success("Sistema de IA carregat correctament!")


st.title("Detector de Fraus Bancaris (TabNet)")
st.markdown("Introdueix les dades de la transacció per analitzar-ne el risc.")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        amt = st.number_input("Import de la transacció ($)", min_value=0.0, value=100.0)
        gender = st.selectbox("Gènere", ["M", "F"])
        category = st.selectbox("Categoria de la despesa", [
            'misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 'misc_pos',
            'grocery_net', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care',
            'health_fitness', 'travel', 'kids_pets', 'home'
        ])
        
    with col2:
        state = st.text_input("Estat (Codi de 2 lletres, ex: NY, CA)", "NY", max_chars=2)
        trans_date = st.date_input("Data de la transacció", datetime.now())
        trans_time = st.time_input("Hora de la transacció", datetime.now())
        dob = st.date_input("Data de naixement (DOB)", datetime(1980, 1, 1))

    submitted = st.form_submit_button("Analitzar Transacció")

if submitted:
    
    dt_trans = datetime.combine(trans_date, trans_time)
    trans_ts = int(time.mktime(dt_trans.timetuple()))
    
    dt_dob = datetime.combine(dob, datetime.min.time())
    dob_ts = int(time.mktime(dt_dob.timetuple())) 
    
    input_data = pd.DataFrame({
        'amt': [amt],
        'gender': [gender],
        'category': [category],
        'state': [state],
        'trans_date_trans_time': [trans_ts],
        'dob': [dob_ts]
    })
    
    input_data = pd.get_dummies(input_data, columns=['gender', 'category', 'state'])
    

    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    input_scaled = scaler.transform(input_data.values)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"FRAU DETECTAT! (Probabilitat: {probability:.2%})")
        st.markdown("Es recomana bloquejar aquesta transacció immediatament.")
    else:
        st.balloons()
        st.success(f"Transacció LEGÍTIMA (Risc: {probability:.2%})")