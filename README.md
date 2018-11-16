# Machine Learning - Classification automatique d'images

Le Cerema Méditerranée explore des techniques de classification automatique d’images (par machine learning) pour déterminer la probabilité de présence (bâti/non-bâti) de bâtiments sur des ortho-photographies, par apprentissage supervisé.
L’objectif de cet exercice est de présenter un cas concret d’utilisation des techniques d’intelligence artificielle dans une application métier.

![alt tag](https://user-images.githubusercontent.com/19548578/48620419-68609780-e9a0-11e8-9125-da7b59e8cf9d.png)

# Pré-requis

Dans notre cas, nous avons été amenés à procéder à l’installation de différents éléments (étape non développée dans cet article) :
- python 3.6
- keras, tensorflow
- QGIS (pour la génération des lots d'entraînement)
- création d'une bibliothèque d'images pour entraîner le modèle
Ne disposant pas de catalogue d'images directement disponible sur internet (il en existe différents pour identifier chats/chiens, modes de transports...), nous avons réalisé notre propre catalogue d'image grâce à la fonctionnalité Atlas de QGIS à partir de données suivantes: une ortho-photographie + la couche "bâtiments" de la BDTopo.
   - réalisation d'un dallage de 50m x 50m
   - croisement géographique avec les bâtiments BDTopo pour catégoriser les dalles en 2 classes:
--> des dalles "bati"    


