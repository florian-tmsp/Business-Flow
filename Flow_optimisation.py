# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:20:55 2020

@author: Flo
"""

import numpy as np


# deux variable à initialiser
#alpha -> learning rate

gamma = 0.75
alpha = 0.9


# 1- Definir l'environnement

#Dictionnaire d'état
# faire une relation état à clé
position_to_etat = {'A': 0,
                     'B': 1,
                     'C': 2, 
                     'D': 3, 
                     'E': 4,
                     'F': 5, 
                     'G': 6, 
                     'H': 7, 
                     'I': 8, 
                     'J': 9, 
                     'K': 10,
                     'L': 11}


#Définir la récompense (Markov)
#Matrice de déplacemnt (Théorie des graphs)

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
               [1,0,1,0,0,1,0,0,0,0,0,0],
               [0,1,0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,0,0,1,0,0,0,0],
               [0,0,0,0,0,0,0,0,1,0,0,0],
               [0,1,0,0,0,0,0,0,0,1,0,0],
               [0,0,1,0,0,0,1000,1,0,0,0,0],
               [0,0,0,1,0,0,1,0,0,0,0,1],
               [0,0,0,0,1,0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0,0,1,0,1,0],
               [0,0,0,0,0,0,0,0,0,1,0,1],
               [0,0,0,0,0,0,0,1,0,0,1,0]])

print(R)

#algo Q- learning
    
#init Q-Values

Q = np.array(np.zeros([12,12]))



print(Q)
print(R.shape[1])

# Définition de la foncrion de difference temporelle
#https://en.wikipedia.org/wiki/Temporal_difference_learning

for i in range (0,1000):
    etat_t = np.random.randint(0, 12) #exclusion de la bande supérieure -> +1
    mouvement_possible = []
    for k in range (R.shape[1]):
        if R[etat_t, k] > 0: #création de la liste des mouvement possible
            mouvement_possible.append(k) #stockage des choix possible dans une liste
    etat_suivant = np.random.choice(mouvement_possible) #choix aléatoire dans la liste
    DT = R[etat_t,  etat_suivant] + gamma * Q[etat_suivant, np.argmax(Q[etat_suivant,])] - Q[etat_t, etat_suivant]
    Q[etat_t, etat_suivant] += alpha * DT




#matrice des routes prioritaires : sélectionner la plus hautes valeurs

print(Q.astype(int)) 


#passer de l'état à la position // inversion du dictionnaire initial
etat_to_position = {etat : position for position, etat in position_to_etat.items()}

print(etat_to_position)  


etat_to_position = {etat : position for position, etat in position_to_etat.items()}

print(etat_to_position)

def route(position_depart, position_finale):
    route = [position_depart] #initialisation de la route qui commence à la position de départ
    position_suivante = position_depart #initialisation de la position
    while (position_suivante != position_finale): #tant que la position cible n'est pas atteinte
        etat_depart = position_to_etat[position_depart] #prend dans le dictionnaire
        etat_suivant = np.argmax(Q[etat_depart,])# prend la valeur la plus élevé dans la matrice des routes
        position_suivante = etat_to_position[etat_suivant]#donne la position suivante en passant par le dictionnaire i
        route.append(position_suivante)
        position_depart = position_suivante
    return route

#test

print('Route:')
route('E', 'G') 