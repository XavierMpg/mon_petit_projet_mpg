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

# In[8]:

# Page d'accueil
def home():
    st.title("Mon Petit Accueil")
    st.image("Titre.png", width=300)
    st.header("Mon Petit Gazon")
    st.write("Application de « fantasy football ». Jeu français qui s'appuie sur six compétitions européennes. L'objectif du jeu est de gagner le championnat de la ligue créée avec des matchs aller-retour. Le jeu se déroule en plusieurs étapes : création de la ligue, recrutement des équipes lors d'un mercato, championnat avec confrontations lors des matchs des journées réelles de championnat.")
    st.caption("Cliquez sur Mon Petit Menu Déroulant en haut à gauche du terrain")

# Page Nous
def equipe():
    st.subheader("Ma Petite Equipe")
    st.header("Xavier Zinedine Girard Zidane")
    st.header("Delphine Pep Belly Guardiola")
    st.image("https://github.com/XavierMpg/mon_petit_projet_mpg/blob/9db970e1b261b752f0353e15d4145a14641dd311/ZZ%20PG.png", width=300)
        
# Page Source de Données MPGStats
def source():
    st.title("Ma Petite Source")
    st.write("Accès au site Source de nos données MPGstats")
    if st.button('Ma Petite Source de Données'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/')
    st.image("https://github.com/XavierMpg/mon_petit_projet_mpg/blob/9db970e1b261b752f0353e15d4145a14641dd311/MPG%20Stade.png", width=300)
    
# Page Echantillon Dataset
def echantillon():
    st.title("Mon Petit Echantillon")
    #st.image("C:\Users\xavie\Documents\assets\MPG confettis.png", width=300)
    csv_path = '/Users/xavie/Documents/dataset_mpg.csv'
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
    
def train():
    st.title("Mon Petit Entrainement")
    
    # Définir les options du menu déroulant
    options = ["Mes Petites Datas", "Ma Petite Préparation", "Ma Petite Prédiction", "Mes Petites Pépites"]

    # Créer le menu déroulant avec la fonction selectbox de Streamlit
    selection = st.selectbox("Sélectionnez une option", options)
    
    if selection == "Mes Petites Datas":
        data()  # Appeler la fonction data
    elif selection == "Ma Petite Préparation":
        prepa()
    elif selection == "Ma Petite Prédiction":
        entrain()  
    elif selection == "Mes Petites Pepites":
        pepite()

# Sous Menu Mes Petites Datas
def data():
    st.header("Mes Petites Datas")
    st.write("Téléchargement des datas à jour")
    if st.button('LIGUE 1'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Ligue-1/custom')
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/ligue 1.png", width=100)
    if st.button('PREMIER LEAGUE'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Premier-League/custom')
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/premier league.png", width=100)
    if st.button('LIGA'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Liga/custom')
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/liga.png", width=100)
    if st.button('SERIE A'):
        webbrowser.open_new_tab('https://www.mpgstats.fr/top/Serie-A/custom')
    st.image("/Users/benbastian/Desktop/Streamlit MPG/assets/serie A.png", width=100)

# Sous Menu Ma Petite Préparation
def prepa():
    st.header("Ma Petite Préparation")
    
    '''
    def average_mean(df,journée_depart, num_journée=4):
        return df[[f'j{journée_depart-day}' for day in range(num_journée)]].sum(axis=1)/np.count_nonzero(df[[f'j{journée_depart-day}' for day in range(num_journée)]], axis=1)

    def concat_journées(list_days,intervalle=4):
        list_dfs = []
    #on parcourt toutes les saisons
        for journee in list_days:
        #numéro de la saison : par exemple pour PL J19.xlsx ce sera 19
        #on utilise un regex pour extraire le chiffre ; [0-9] = tout chiffre entre 0 et 9 , {1,2} = présents une ou deux fois (on pourrait mettre 4 si les noms de fichiers étaient du style 2018 , 2019 etc)
            num_journée = int(re.search(r'[0-9]{1,2}',journee).group(0)) 
        #on ouvre le fichier avec header = 2
            df = pd.read_excel(journee,header=2)
        
        df["moy_j"]= average_mean(df,num_journée,intervalle)
        df["moy_j_10"]= average_mean(df,num_journée,10)
        #on remplit valeur manquantes par '0'
        df = df.fillna('0')
        #renommage des colonnes
        df.rename(columns={'%Titu': 'Titu',
                   '%Titu série': 'Titu série',
                   '%Titu 1 an': 'Titu 1 an',
                   '%Passes': 'Passes',
                   '%Duel': 'Duel',
                   '%Win+12J': 'Win12J',
                   '%Win+16J': 'Win16J',
                   '%Win+20J': 'Win20J'}, inplace=True)
         # On remplace j29 j28 j27 ... j1 j38 j37 etc par j-1 j-2 j-3 afin d'uniformiser colonnes selon les journées de reference
        df[[f'j-{i+1}' for i in range(38)] ]= df.iloc[:,[29+i for i in range(38)]].copy()
        #par exemple df[j29] devient df[j-1] etc
        
        #transforme les données non numériques en float 
        df['Cote'] = df['Cote'].astype('float64')
        df['Enchère moy'] = df['Enchère moy'].astype('float64')
        df['Cleansheet'] = df['Cleansheet'].astype('float64')
        df['But/Peno'] = df['But/Peno'].astype('float64')
        df['But/Coup-franc'] = df['But/Coup-franc'].astype('float64')
        df['But/surface'] = df['But/surface'].astype('float64')
        df['Pass decis.'] = df['Pass decis.'].astype('float64')
        df['Passes'] = df['Passes'].astype('float64')
        df['Ballons'] = df['Ballons'].astype('float64')
        df['Interceptions'] = df['Interceptions'].astype('float64')
        df['Tacles'] = df['Tacles'].astype('float64')
        df['Duel'] = df['Duel'].astype('float64')
        df['Fautes'] = df['Fautes'].astype('float64')
        df['But évité'] = df['But évité'].astype('float64')
        df['Min/But'] = df['Min/But'].astype('float64')
        df['Min/But 1 an'] = df['Min/But 1 an'].astype('float64')
        df['Min note/but'] = df['Min note/but'].astype('float64')
        df['Action stoppée'] = df['Action stoppée'].astype('float64')
        df['Occas° créée'] = df['Occas° créée'].astype('float64')
        df['Corner gagné'] = df['Corner gagné'].astype('float64')
        df['moy_j'] = df['moy_j'].astype('float64')
        
        #supprime les colonnes non utiles 
        df = df.drop(['Club', 'Prochain opposant', 'Date', 'Victoire probable', 'Win12J', 'Win16J', 'Win20J'], axis = 1) 
        # on supprime les anciennes colonnesde notes par journées
        df = df.drop(columns=list(df.columns[29:(29+38)]))   
        list_dfs.append(df)
    #on concatene toutes les saisons :
    df_final = pd.concat(list_dfs).reset_index(drop=True)
    return df_final.drop(columns=["Dispo@MPGLaurent?"])

    df = concat_journées(["IT J31.xlsx", "ES J31.xlsx","FR J31.xlsx", "PL  J31.xlsx", "PL J29.xlsx","FR J29.xlsx","ES J27.xlsx","IT J27.xlsx", "IT J15.xlsx", "LF J17.xlsx", "PL J23.xlsx"])
    
    
    def count_titularisations(liste_journées):
        return np.count_nonzero (liste_journées)
# ici on compte titularisations sur 38 journées 

df.apply(lambda x : count_titularisations([ x[f'j-{i+1}'] for i in range(38)]),axis=1)

df['Titu_4'] = df.apply(lambda x : count_titularisations([ x[f'j-{i+1}'] for i in range(4)]),axis=1)

df['Titu_10'] = df.apply(lambda x : count_titularisations([ x[f'j-{i+1}'] for i in range(10)]),axis=1)

#SEPARATION DU JEU DE DONNEES POUR ENTRAINEMENT DES MODELES

#DF JOUEURS OFFENSIFS

df_attack = df.loc[(df['Poste'] == 'A')| (df['Poste'] == 'MO')].reset_index()

#DF JOUEURS DEFENSIFS

df_defense = df.loc[(df['Poste'] == 'MD')|(df['Poste'] == 'DC')|(df['Poste'] == 'DL')].reset_index()

#DF GARDIEN

df_goalkeeper = df.loc[(df['Poste'] == 'G')].reset_index()

# export df_attack au format csv
df_attack.to_csv('df_attack_mpg_v210423.csv')

# export df_defense au format csv
df_defense.to_csv('df_defense_mpg_v210423.csv')

# export df_defense au format csv
df_goalkeeper.to_csv('df_goalkeeper_mpg_v210423.csv')
    
    '''
    
# Sous Menu Ma Petite Prédiction
def entrain():
    st.header("Ma Petite Prédiction")
# Afficher les boutons pour chaque groupe de joueur
    gk_button = st.button("Gardiens de but")
    att_button = st.button("Attaquants")
    def_button = st.button("Défenseurs")

# Si le bouton des gardiens de but est cliqué
    if gk_button:
# Charger les dataframes sur lesquels appliquer le modèle
        df_gk = pd.read_csv('/Users/benbastian/Desktop/df_gk_to_upload.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('/Users/benbastian/Desktop/model_gk.pkl', 'rb') as f:
            model_gk = pickle.load(f)

        predictions_gk = model_gk.predict(df_gk)

        df_gk_output = pd.read_csv('/Users/benbastian/Desktop/df_goalkeeper_mpg_v210423.csv',index_col=0)

        df_gk_output['cote_predite'] = predictions_gk
        df_gk_output['+/- value'] = df_gk_output['cote_predite'] - df_gk_output['Cote']

# output base de données gardien avec cote
        st.write(df_gk_output[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value']])

# Si le bouton des attaquants est cliqué
    if att_button:

# Charger les dataframes sur lesquels appliquer le modèle
        df_att = pd.read_csv('/Users/benbastian/Desktop/df_att_to_upload.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('/Users/benbastian/Desktop/model_att.pkl', 'rb') as f:
            model_att = pickle.load(f)

        predictions_att = model_att.predict(df_att)

        df_att_output = pd.read_csv('/Users/benbastian/Desktop/df_attack_mpg_v210423.csv',index_col=0)

        df_att_output['cote_predite'] = predictions_att
        df_att_output['+/- value'] = df_att_output['cote_predite'] - df_att_output['Cote']

# output base de données attaque avec cote
        st.write(df_att_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value']])

# Si le bouton des défenseurs est cliqué
    if def_button:
# Charger les dataframes sur lesquels appliquer le modèle
        df_def = pd.read_csv('/Users/benbastian/Desktop/df_def_to_upload.csv',index_col=0)

# Charger le modèle à partir du fichier.pkl
        with open('/Users/benbastian/Desktop/model_def.pkl', 'rb') as f:
            model_def = pickle.load(f)

        predictions_def = model_def.predict(df_def)

        df_def_output = pd.read_csv('/Users/benbastian/Desktop/df_defense_mpg_v210423.csv',index_col=0)

        df_def_output['cote_predite'] = predictions_def
        df_def_output['+/- value'] = df_def_output['cote_predite'] - df_def_output['Cote']
        
        # output base de données attaque avec cote
        st.write(df_def_output[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value']])

#Sous Menu Pepite

def pepite():
    st.header("Mes Petites Pépites")
    
    # Afficher les boutons pour chaque groupe de joueur
    gk_button = st.button("Meilleures Côtes Gardiens de but")
    att_button = st.button("Meilleures Côtes Attaquants")
    def_button = st.button("Meilleures Côtes Défenseurs")

    # Si le bouton des gardiens de but est cliqué
    if gk_button:
        df_gk_output = pd.read_csv('/Users/benbastian/Desktop/df_goalkeeper_mpg_v210423.csv',index_col=0)
        df_gk_output['cote_predite'] = predictions_gk
        df_gk_output['+/- value'] = df_gk_output['cote_predite'] - df_gk_output['Cote']
        # Afficher les 5 meilleurs gardiens de but
        st.write("Les 5 meilleurs gardiens de but :")
        st.write(df_gk_output[['Joueur', 'Poste', 'Cote', 'cote_predite', '+/- value']].sort_values('cote_predite', ascending=False).head(5))

    # Si le bouton des attaquants est cliqué
    if att_button:
        df_att_output = pd.read_csv('/Users/benbastian/Desktop/df_attack_mpg_v210423.csv',index_col=0)
        df_att_output['cote_predite'] = predictions_att
        df_att_output['+/- value'] = df_att_output['cote_predite'] - df_att_output['Cote']
        # Afficher les 5 meilleurs attaquants
        st.write("Les 5 meilleurs attaquants :")
        st.write(df_att_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value']].sort_values('cote_predite', ascending=False).head(5))

    # Si le bouton des défenseurs est cliqué
    if def_button:
        df_def_output = pd.read_csv('/Users/benbastian/Desktop/df_defense_mpg_v210423.csv',index_col=0)
        df_def_output['cote_predite'] = predictions_def
        df_def_output['+/- value'] = df_def_output['cote_predite'] - df_def_output['Cote']
        # Afficher les 5 meilleurs attaquants
        st.write("Les 5 meilleurs attaquants :")
        st.write(df_def_output[['Joueur', 'Poste','Cote', 'cote_predite', '+/- value']].sort_values('cote_predite', ascending=False).head(5))
    

# Menu déroulant
menu = ['Mon Petit Accueil', 'Ma Petite Equipe', 'Ma Petite Source', 'Mon Petit Echantillon', "Ma Petite Corrélation", "Mes Petites DataViz", "Mes Petits Joueurs", "Mon Petit Entrainement"]
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
    

   



