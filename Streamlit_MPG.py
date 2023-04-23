#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# In[8]:

# Page d'accueil
def home():
    st.title("Mon Petit Accueil")
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/Titre.png", width=300)
    st.header("Mon Petit Gazon")
    st.write("Application de « fantasy football ». Jeu français qui s'appuie sur six compétitions européennes. L'objectif du jeu est de gagner le championnat de la ligue créée avec des matchs aller-retour. Le jeu se déroule en plusieurs étapes : création de la ligue, recrutement des équipes lors d'un mercato, championnat avec confrontations lors des matchs des journées réelles de championnat.")
    st.caption("Cliquez sur Mon Petit Menu Déroulant en haut à gauche du terrain")

# Page Nous
def equipe():
    st.subheader("Ma Petite Equipe")
    st.header("Xavier Zinedine Girard Zidane")
    st.header("Delphine Pep Belly Guardiola")
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/ZZ PG.png", width=300)
        
# Page Source de Données MPGStats
def source():
    st.title("Ma Petite Source")
    st.write("Accès au site Source de nos données MPGstats")
    if st.button('Ma Petite Source de Données'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/')
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/MPG Stade.png", width=300)
    
# Page Echantillon Dataset
def echantillon():
    st.title("Mon Petit Echantillon")
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/MPG confettis.png", width=300)
    csv_path = '/Users/benbastian/Desktop/dataset_mpg.csv'
    if st.button('Charger Mon Petit Exemple de Dataset MPG'):
        df = pd.read_csv(csv_path)
        st.write(df)

# Page Heatmap
def heatmap():
    datadef = pd.read_csv('/Users/benbastian/Desktop/df_def_output_model_v150423.csv')
    dataoff = pd.read_csv('/Users/benbastian/Desktop/df_attack_output_model_v150423 (2).csv')
    datagoal = pd.read_csv('/Users/benbastian/Desktop/df_goalkeeper_output_model_v150423 (1).csv')
    datafull = pd.concat([datadef, dataoff,datagoal])
    st.title("Ma Petite Corrélation")
    cols = ['Cote','Enchère moy', 'Note','Note série','Note 1 an', 'Nb match', 'Nb match série', 'Nb match 1 an', 'Variation', 'Var série', 'Var 1 an', 'But', 'Buts série', 'Buts 1 an', 'Titu', 'Titu série', 'Titu 1 an', 'Temps', 'Tps série', 'Tps 1 an', 'Tps moy', 'Tps moy série', 'Tps moy 1 an', 'Min/But', 'Min/But 1 an', 'Min note/but', 'Prix/but', 'Cleansheet', 'But/Peno', 'But/Coup-franc', 'But/surface', 'Pass decis.','Corner gagné', 'Passes', 'Ballons','Interceptions', 'Tacles', 'Duel', 'Fautes', 'But évité', 'Action stoppée', 'moy_j_10','Titu_4', 'Titu_10']
    data = datafull[cols]
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)

# Display the heatmap in Streamlit
    st.pyplot(fig)
    
# Page Visualisation de la Cote en fonction de variables choisies
def dataviz():
    st.title("Mes Petites DataViz")
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/Zindine.png", width=300)
    datadef = pd.read_csv('/Users/benbastian/Desktop/df_def_output_model_v150423.csv')
    dataoff = pd.read_csv('/Users/benbastian/Desktop/df_attack_output_model_v150423 (2).csv')
    datagoal = pd.read_csv('/Users/benbastian/Desktop/df_goalkeeper_output_model_v150423 (1).csv')
    datafull = pd.concat([datadef, dataoff,datagoal])
    
    variable = st.selectbox("Choisissez la variable à mettre en perspective avec la Cote", ["Note", "Nb match", "Nb match 1 an", "But", "Buts 1 an", "Titu", "Titu 1 an", "Tps moy", "Pass decis.", "Occas° créée", "Passes", "moy_j", "moy_j_10"])
    fig = px.scatter(datafull, x="Cote", y=variable, trendline="ols")
    st.plotly_chart(fig)
        
# Page Data des Joueurs Trois sections
def visualisation():
    st.title("Mes Petits Gardiens")
    data = pd.read_csv('/Users/benbastian/Desktop/df_goalkeeper_output_model_v150423 (1).csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur = st.selectbox("Sélectionnez un joueur", joueurs)
    joueur_data = data[data['Joueur'] == selected_joueur]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)
    
    st.title("Mes Petits Joueurs Défensifs")
    data = pd.read_csv('/Users/benbastian/Desktop/df_def_output_model_v150423.csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur = st.selectbox("Sélectionnez un joueur", joueurs)
    joueur_data = data[data['Joueur'] == selected_joueur]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)

    st.title("Mes Petits Joueurs Offensifs")
    data = pd.read_csv('/Users/benbastian/Desktop/df_attack_output_model_v150423 (2).csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur = st.selectbox("Sélectionnez un joueur", joueurs)
    joueur_data = data[data['Joueur'] == selected_joueur]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)

# Menu déroulant
menu = ['Mon Petit Accueil', 'Ma Petite Equipe', 'Ma Petite Source', 'Mon Petit Echantillon', "Ma Petite Corrélation", "Mes Petites DataViz", "Mes Petits Joueurs"]
choice = st.sidebar.selectbox("Mon Petit Menu Déroulant", menu)

# Affichage de la page en fonction du choix dans le menu déroulant
if choice == 'Mon Petit Accueil':
    home()
elif choice == 'Ma Petite Equipe':
    equipe()
elif choice == "Ma Petite Source":
    source()
elif choice == "Mon Petit Echantillon":
    echantillon()
elif choice == "Ma Petite Corrélation":
    heatmap()    
elif choice == "Mes Petites DataViz":
    dataviz()
elif choice == "Mes Petits Joueurs":
    visualisation()
    

   



