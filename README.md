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
![alt tag](https://user-images.githubusercontent.com/19548578/48621056-9f37ad00-e9a2-11e8-8feb-0e854ffa0cbe.png)

--> et des dalles "non-bati"
![alt tag](https://user-images.githubusercontent.com/19548578/48620419-68609780-e9a0-11e8-9125-da7b59e8cf9d.png)

La requête que nous avons utilisée est très simple (peut-être trop) car elle se contente de dire si un bâti (ou une portion de ce dernier) est présent sur le carreau. En réalité, pour avoir un lot de meilleure qualité, un petit nettoyage serait à réaliser afin de supprimer:
- les détections abusives/manquantes (dues à un écart d'actualité entre l'ortho-photo et la couche de bâtiments)
- les dalles où des bâtiments sont réellement présents au sol mais, soit invisibles sur la photo depuis le ciel (à cause du couvert végétal), soit représentant une trop petite surface (d'où la possibilité de mettre des seuils minimums lors de la catégorisation établie précédemment)

Nous avons établi une bibliothèque comprenant au total 12100 images: 10000 pour entraîner le modèle , 2000 pour le valider, et 100 pour les tests.

