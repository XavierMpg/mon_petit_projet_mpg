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
import pickle
import hashlib

# In[8]:

# Page d'accueil
def home():
    st.write('<p style="color:blue;font-size: 40px;"><b>Mon Petit Accueil</b></p>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/bd5487c072c4c5c018619d277e725ea2e870feae/Titre.png", width=300)
    st.write("Application de « fantasy football ». Jeu français qui s'appuie sur six compétitions européennes. L'objectif du jeu est de gagner le championnat de la ligue créée avec des matchs aller-retour. Le jeu se déroule en plusieurs étapes : création de la ligue, recrutement des équipes lors d'un mercato, championnat avec confrontations lors des matchs des journées réelles de championnat.")
    st.caption("Cliquez sur Mon Petit Menu Déroulant en haut à gauche du terrain")

# Page Nous
def equipe():
    st.write('<p style="color:red;font-size: 40px;"><b>Ma Petite Equipe</b></p>', 
unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/ZZ%20PG.png", width=300)
    
    col1, col2 = st.columns(2)
    
    col1.subheader("Xavier Zinedine Girard Zidane")
    if col1.button('Le Petit Linkedin de Xavier Zidane'):
        webbrowser.open_new_tab('https://www.linkedin.com/in/xavier-girard-3200046')    
                
    col2.subheader("Delphine Pep Belly Guardiola")
    if col2.button('Le Petit Linkedin de Delphine Guardiola'):
        webbrowser.open_new_tab('https://www.linkedin.com/in/delphine-s-7b2932253')    
        
# Page Source de Données MPGStats
def source():
    st.write('<p style="color:blue;font-size: 40px;"><b>Ma Petite Source</b></p>', unsafe_allow_html=True)
    st.write("Accès au site Source de nos données MPGstats")
    if st.button('Ma Petite Source de Données'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/')
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/MPG%20Stade.png", width=300)
    
# Page Echantillon Dataset
def echantillon():
    st.write('<p style="color:red;font-size: 40px;"><b>Mon Petit Echantillon</b></p>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/MPG%20confettis.png", width=300)
    csv_path = 'https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/dataset_mpg.csv'
    if st.button('Charger Mon Petit Exemple de Dataset MPG'):
        df = pd.read_csv(csv_path, index_col = 0)
        st.write(df)

# Page Heatmap
def heatmap():
    datadef = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_defense_output_v210423.csv')
    dataoff = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_attack_output_v210423.csv')
    datagoal = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_goalkeeper_output_v210423.csv')
    datafull = pd.concat([datadef, dataoff,datagoal])
    st.title("Ma Petite Corrélation")
    cols = ['Cote','Enchère moy', 'Note','Note série','Note 1 an', 'Nb match', 'Nb match série', 'Nb match 1 an', 'Variation', 'Var série', 'Var 1 an', 'But', 'Buts série', 'Buts 1 an', 'Titu', 'Titu série', 'Titu 1 an', 'Temps', 'Tps série', 'Tps 1 an', 'Tps moy', 'Tps moy série', 'Tps moy 1 an', 'Min/But', 'Min/But 1 an', 'Min note/but', 'Prix/but', 'Cleansheet', 'But/Peno', 'But/Coup-franc', 'But/surface', 'Pass decis.','Corner gagné', 'Passes', 'Ballons','Interceptions', 'Tacles', 'Duel', 'Fautes', 'But évité', 'Action stoppée', 'moy_j_10','Titu_4', 'Titu_10']
    top_10_corr = datafull.corr()['Cote'].abs().nlargest(10).index
    data = datafull[top_10_corr]
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)

# Display the heatmap in Streamlit
    st.pyplot(fig)
    
# Page Visualisation de la Cote en fonction de variables choisies
def dataviz():
    st.write('<p style="color:red;font-size: 40px;"><b>Mes Petites DataViz</b></p>', unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/Zindine.png", width=300)
    datadef = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_defense_output_v210423.csv')
    dataoff = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_attack_output_v210423.csv')
    datagoal = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_goalkeeper_output_v210423.csv')
    datafull = pd.concat([datadef, dataoff,datagoal])
    
    variable = st.selectbox("Choisissez la variable à mettre en perspective avec la Cote", ["Note", "Nb match", "Nb match 1 an", "But", "Buts 1 an", "Titu", "Titu 1 an", "Tps moy", "Pass decis.", "Occas° créée", "Passes", "moy_j", "moy_j_10"])
    fig = px.scatter(datafull, x="Cote", y=variable, trendline="ols")
    st.plotly_chart(fig)
        
# Page Data des Joueurs Trois sections
def visualisation():
    st.title("Mes Petits Gardiens")
    data = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_goalkeeper_output_v250423.csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur_gk = st.selectbox("Sélectionnez un joueur", joueurs, key=hashlib.md5("joueur_gk".encode()).hexdigest())
    joueur_data = data[data['Joueur'] == selected_joueur_gk]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)
    
    st.title("Mes Petits Joueurs Défensifs")
    data = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_defense_output_v250423.csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur_def = st.selectbox("Sélectionnez un joueur", joueurs, key=hashlib.md5("joueur_def".encode()).hexdigest())
    joueur_data = data[data['Joueur'] == selected_joueur_def]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)

    st.title("Mes Petits Joueurs Offensifs")
    data = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_attack_output_v250423.csv')
    joueurs = list(data['Joueur'].unique())
    selected_joueur_att = st.selectbox("Sélectionnez un joueur", joueurs, key=hashlib.md5("joueur_att".encode()).hexdigest())
    joueur_data = data[data['Joueur'] == selected_joueur_att]
   
    # Sélection des variables pour le joueur sélectionné
    selected_variables = joueur_data[['Cote', 'Enchère moy', 'Note', 'Nb match', 'Nb match 1 an', 'But', 'Buts 1 an', 'Titu', 'Titu 1 an', 'Tps 1 an', 'Tps moy', 'Min/But', 'Min/But 1 an', 'But/Peno', 'But/Coup-franc', 'But/surface','Pass decis.', 'Occas° créée', 'Passes', 'moy_j', 'moy_j_10']]
    st.write(selected_variables)
    
def train():
    st.title("Mon Petit Entrainement")
    
    # Définir les options du menu déroulant
    options = ["Mes Petites Datas", "Mes Petits Modèles", "Ma Petite Prédiction"]

    # Créer le menu déroulant avec la fonction selectbox de Streamlit
    selection = st.selectbox("Sélectionnez une option", options)
    
    if selection == "Mes Petites Datas":
        data()  # Appeler la fonction data
    elif selection == "Mes Petits Modèles":
        model()  
    elif selection == "Ma Petite Prédiction":
        entrain()
            
             
# Sous Menu Mes Petites Datas
def data():
    st.header("Mes Petites Datas")
    st.write("Téléchargement des datas à jour")
    if st.button('LIGUE 1'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Ligue-1/custom')
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/ligue%201.png", width=100)
    if st.button('PREMIER LEAGUE'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Premier-League/custom')
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/premier%20league.png", width=100)
    if st.button('LIGA'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Liga/custom')
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/liga.png", width=100)
    if st.button('SERIE A'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Serie-A/custom')
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/serie%20A.png", width=100)

# Sous Menu Modèles
def model():
    st.write('<p style="color:red;font-size: 40px;"><b>Mon Petit Modèle</b></p>', unsafe_allow_html=True)
    
    st.subheader("Modèle Gardien")
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/ExtrasTreesGardien.png", width=500) 
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/Feature%20Importance%20Gardien.png", width=500) 
                
    st.subheader("Modèle Offensifs")
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/ExtrasTreesOffensifs.png", width=500) 
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/Feature%20Importance%20Offensifs.png", width=500)
    
    st.subheader("Modèle Defensifs")
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/ExtrasTreesDefensifs.png", width=500) 
    st.image("https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/Feature%20Importance%20Defensifs.png", width=500)
             
# Sous Menu Ma Petite Prédiction
def entrain():
    st.header("Ma Petite Prédiction")  
    
# Ajouter un widget Range Slider pour filtrer le dataframe sur la variable 'Cote'
    cote_range = st.slider('Filtrer sur la variable Cote', 0 , 60 , (0, 60))
    min_value = cote_range[0]
    max_value = cote_range[1]
    
    
# Afficher les boutons pour chaque groupe de joueur
    gk_button = st.button("Gardiens de but")
    att_button = st.button("Attaquants")
    def_button = st.button("Défenseurs")

# Si le bouton des gardiens de but est cliqué
    if gk_button:
# Charger les dataframes sur lesquels appliquer le modèle
        df_gk = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_gk_to_upload_v250423.csv',index_col=0)
  
# Charger le modèle à partir du fichier.pkl
        with open('model_gk.pkl', 'rb') as f:
            model_gk = pickle.load(f)


# Appliquer le modèle sur le dataframe filtré
        predictions_gk = model_gk.predict(df_gk)

# Chargement du dataset de base        
        df_gk_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_goalkeeper_mpg_v250423.csv',index_col=0)
   
# Ajout au dataset de base de la Cote prédite et de la plus value
        df_gk_output['cote_predite'] = predictions_gk
        df_gk_output['+/- value'] = df_gk_output['cote_predite'] - df_gk_output['Cote']

# Filtrer le dataframe sur la variable 'cote'
        df_gk_output = df_gk_output[(df_gk_output['Cote'] >= min_value) & (df_gk_output['Cote'] <= max_value)]
        
# output base de données gardien avec cote
        st.write(df_gk_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value', 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])

# Si le bouton des attaquants est cliqué
    if att_button:

# Charger les dataframes sur lesquels appliquer le modèle
        df_att = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_att_to_upload_v250423.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('model_att.pkl', 'rb') as f:
            model_att = pickle.load(f)

# Appliquer le modèle sur le dataframe filtré
        predictions_att = model_att.predict(df_att_filtered)
    
 # Ouverture du dataset de base
        df_att_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_attack_mpg_v250423.csv',index_col=0)

 # Ajout de la Cote prédite et de la plus value
        df_att_output['cote_predite'] = predictions_att
        df_att_output['+/- value'] = df_att_output['cote_predite'] - df_att_output['Cote']

# Filtrer le dataframe sur la variable 'cote'
        df_att_output = df_att_output[(df_att_output['Cote'] >= min_value) & (df_att_output['Cote'] <= max_value)]
        
# output base de données attaque avec cote
        st.write(df_att_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value', 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])

# Si le bouton des défenseurs est cliqué
    if def_button:
# Charger les dataframes sur lesquels appliquer le modèle
        df_def = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_def_to_upload_v250423.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('model_def.pkl', 'rb') as f:
            model_def = pickle.load(f)


# Appliquer le modèle sur le dataframe filtré
        predictions_def = model_att.predict(df_def_filtered)

    
 # Ouverture du dataset de base
        df_def_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_defense_mpg_v250423.csv',index_col=0)

        df_def_output['cote_predite'] = predictions_def
        df_def_output['+/- value'] = df_def_output['cote_predite'] - df_def_output['Cote']
 
# Filtrer le dataframe sur la variable 'cote'
        df_def_output = df_def_output[(df_def_output['Cote'] >= min_value) & (df_def_output['Cote'] <= max_value)]
    
 # output base de données attaque avec cote
        st.write(df_def_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value', 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])

#Sous Menu Pepite

def pepite():
    st.header("Mes Petites Pépites")
    
    # Afficher les boutons pour chaque groupe de joueur
    gkbest_button = st.button("Meilleures Côtes Gardiens de but")
    attbest_button = st.button("Meilleures Côtes Attaquants")
    defbest_button = st.button("Meilleures Côtes Défenseurs")

    # Si le bouton des gardiens de but est cliqué
    if gkbest_button:
        df_gk = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_gk_to_upload_v250423.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('model_gk.pkl', 'rb') as f:
             model_gk = pickle.load(f)

        predictions_gk = model_gk.predict(df_gk)
        
        df_gk_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_goalkeeper_mpg_v250423.csv',index_col=0)
        df_gk_output['cote_predite'] = predictions_gk
        df_gk_output['+/- value'] = df_gk_output['cote_predite'] - df_gk_output['Cote']
        top_gk = df_gk_output.loc[df_gk_output['Cote'] <= 5].sort_values(by='+/- value', ascending=False).head(5)
        # Afficher les 5 meilleurs gardiens de but
        st.write("Mes 5 petits gardiens :")
        st.write(top_gk[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value' , 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])

    # Si le bouton des attaquants est cliqué
    if attbest_button:
        df_att = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_att_to_upload_v250423.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl

        with open('model_att.pkl', 'rb') as f:
            model_att = pickle.load(f)

        predictions_att = model_att.predict(df_att)
        df_att_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_attack_mpg_v250423.csv',index_col=0)
        df_att_output['cote_predite'] = predictions_att
        df_att_output['+/- value'] = df_att_output['cote_predite'] - df_att_output['Cote']
        top_att = df_att_output.loc[df_att_output['Cote'] <= 5].sort_values(by='+/- value', ascending=False).head(10) 
        # Afficher les 5 meilleurs attaquants
        st.write("Mes 10 pépites offensives :")
        st.write(top_att[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value' , 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])

    # Si le bouton des défenseurs est cliqué
    if defbest_button:
        df_def = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_def_to_upload_v250423.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('model_def.pkl', 'rb') as f:
            model_def = pickle.load(f)

        predictions_def = model_def.predict(df_def)

        df_def_output = pd.read_csv('https://raw.githubusercontent.com/XavierMpg/mon_petit_projet_mpg/main/df_defense_mpg_v250423.csv',index_col=0)
        df_def_output['cote_predite'] = predictions_def
        df_def_output['+/- value'] = df_def_output['cote_predite'] - df_def_output['Cote']
        top_def = df_def_output.loc[df_def_output['Cote'] <= 5].sort_values(by='+/- value', ascending=False).head(10)
        # Afficher les 5 meilleurs attaquants
        st.write("Mes 10 pépites défensives :")
        st.write(top_def[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value' , 'moy_j' , 'moy_j_10' , 'j-1' , 'j-2' , 'j-3' ,'j-4']])
    

# Menu déroulant
menu = ['Mon Petit Accueil', 'Ma Petite Equipe', 'Ma Petite Source', 'Mon Petit Echantillon', "Ma Petite Corrélation", "Mes Petites DataViz", "Mes Petits Joueurs", "Mon Petit Entrainement", "Mes Petites Pépites"]
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
elif choice == "Mon Petit Entrainement":
    train()
elif choice == "Mes Petites Pépites":
    pepite()
    

   



