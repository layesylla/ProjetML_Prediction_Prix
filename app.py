import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Charger le modèle et les données sauvegardées
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Prédiction du Prix des Laptops")
#Les champ de notre model
# ccampany
company = st.selectbox('Brand', df['Company'].unique())
#type
type_name = st.selectbox('Type', df['TypeName'].unique())
#la RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
#poids
weight = st.number_input('Poids du laptop', min_value=0.0, step=0.1)
#Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
#ips
ips = st.selectbox('IPS', ['No', 'Yes'])
#screen_size
screen_size = st.number_input('Screen Size', min_value=0.1, step=0.1)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160',
                                                 '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024, 2048])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Prédire le prix'):
    # Transformation des entrées binaires en 0/1
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calcul du PPI,la pixelisation
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size if screen_size > 0 else 0

    # Construire la requête sous forme de DataFrame
    query = pd.DataFrame([[company, type_name, ram, weight, touchscreen, ips, ppi, hdd, ssd, cpu, gpu, os]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD',
                                  'Cpu brand', 'Gpu brand', 'os'])

    # Assurer que les colonnes correspondent à celles du pipeline
    for col in pipe.feature_names_in_:
        if col not in query.columns:
            query[col] = 0  # Aon ajoute les colonnes manquantes de notre query ,c'est ce qui nous induisait en erreur plusieirs foi

    # Réordonner les colonnes selon celles du pipeline
    query = query[pipe.feature_names_in_]

    # Faire la prédiction
    prediction = pipe.predict(query)
    st.title(f"Prix estimé : {int(np.exp(prediction[0]))} F")
