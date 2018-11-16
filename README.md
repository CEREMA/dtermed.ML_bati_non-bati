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
![alt tag](https://user-images.githubusercontent.com/19548578/48621084-b37baa00-e9a2-11e8-8888-835b87e1eda3.png)

La requête que nous avons utilisée est très simple (peut-être trop) car elle se contente de dire si un bâti (ou une portion de ce dernier) est présent sur le carreau. En réalité, pour avoir un lot de meilleure qualité, un petit nettoyage serait à réaliser afin de supprimer:
- les détections abusives/manquantes (dues à un écart d'actualité entre l'ortho-photo et la couche de bâtiments)
- les dalles où des bâtiments sont réellement présents au sol mais, soit invisibles sur la photo depuis le ciel (à cause du couvert végétal), soit représentant une trop petite surface (d'où la possibilité de mettre des seuils minimums lors de la catégorisation établie précédemment)

Nous avons établi une bibliothèque comprenant au total 12100 images: 10000 pour entraîner le modèle , 2000 pour le valider, et 100 pour les tests.

![alt tag](https://user-images.githubusercontent.com/19548578/48621256-3a308700-e9a3-11e8-9270-5951afa05b90.png)

# Préparation du modèle

Les différents paramètres du modèle sont définis dans un script python. On y retrouve par exemple:
- une **fonction d'augmentation d'image**: des actions de rotation, décalage, effet miroir, étirement ou zoom permettent d'augmenter significativement le nombre d'images en entrée (sans toucher à la bibliothèque) et d'éliminer certains biais engendrés par l'orientation ou la position de certains éléments au sein de l'image
- la **définition des classes** recherchées, aussi appelée labels ou étiquettes (bati/non-bati)
- l'**emplacement** des données d'entrée
- l'enchaînement des **blocs de fonctions** de notre réseau de neurones

Exemple d'enchaînement:

```model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```

Ce modèle est constitué de 5 blocs, à savoir :
- trois blocs de **convolution-pooling** qui servent à interpréter les images et en extraire les features, soit les motifs caractéristiques, plus ou moins abstraits à mesure que l'on va profondément dans le réseau.
- deux **couches entièrement connectées**, dont la première sert de couche intermédiaire avec la dernière qui sert à la classification (avec un neurone de sortie pour sortie binaire)
La courbe obtenue à l'issue de l'entraînement donne des renseignements sur la pertinence du réseau entraîné, et peut mettre en évidence certains biais comme un sur-apprentissage du réseau de neurones (lorsque le réseau apprend par cœur et n'arrive pas à généraliser ses connaissances à de nouvelles images qu'on lui soumet).

![alt tag](https://user-images.githubusercontent.com/19548578/48621277-47e60c80-e9a3-11e8-8f3e-53b6a780cb11.png)

La visualisation des courbes de précision et de coût, pour les lots d'entraînement et de validation, permet de voir la performance du modèle et le nombre d'epochs à compter duquel le modèle atteint ses meilleures performances.
En général, à la suite de cela, on modifie le nombre de blocs, le type de fonctions à utiliser, la profondeur du réseau de neurones, de façon à obtenir un modèle plus performant. L'atteinte d'un "meilleur" modèle, à savoir un qui ne sur-apprend pas, et qui réponde aussi bien aux données d'entraînement que de validation, se fait par une succession d'itérations et d'essais à la suite desquels on modifie les hyper-paramètres du réseau de neurones.
Cela prend généralement du temps et s'apparente davantage à un art qu'à une science exacte.

# Test du modèle

Le test du modèle est effectué sur 100 nouvelles images, n'ayant servi ni à l'entraînement, ni à la validation, mais qui ont tout de même été catégorisées lors de la création du jeu de données. 
Le résultat est rendu sous la forme d'images annotées avec la prédiction d'appartenance à l'une ou l'autre des 2 classes attendues: bâti / non-bâti
![alt tag](https://user-images.githubusercontent.com/19548578/48621277-47e60c80-e9a3-11e8-8f3e-53b6a780cb11.png)







