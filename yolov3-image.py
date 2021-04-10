import numpy as np
import cv2
import time
 
image_BGR = cv2.imread('images/dog.jpg', cv2.IMREAD_UNCHANGED)
#image_BGR = cv2.imread('images/yolo_image.jpeg')
#image_BGR = cv2.imread('images/women-looking-financial-documents.jpg', cv2.IMREAD_UNCHANGED)
#
#"""
#Réduire la taille de l'image
#"""
#print('Original Dimensions : ',image_BGR.shape)
#
#scale_percent = 50 # percent of original size
#width = int(image_BGR.shape[1] * scale_percent / 90)
#height = int(image_BGR.shape[0] * scale_percent / 100)
#dim = (width, height)
#  
## redimensionner l'image pour reuire la taille de l'image
#resized = cv2.resize(image_BGR, dim, interpolation = cv2.INTER_AREA)
# 
#print('redimensionnellement: ',resized.shape)
# 
#cv2.imshow("image redimensionnee", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#"""
#fin de réduction de l'image
#"""


#Pour afficher l'image, utilisez le code suivant,
cv2.imshow("image originale", image_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

#L'image ci dessus est notre image originale à partir de laquelle on veut détecter autant d'objets que possible. Mais on ne peut pas donner cette image directement à l'algorithme, on doit donc effectuer une conversion à partir de cette image originale. C'est ce qu'on appelle la conversion d'objets blob, qui consiste essentiellement à extraire des fonctionnalités de l'image c'est une manière de prétraitement de l'image.
#
#On va détecter les objets dans le blob via cv2.dnn.blobFromImage et en utilisant quelques variables: image_BGR est le nom du fichier, facteur d'échelle de 1/255, taille de l'image à utiliser dans le blob soit (416,416), le pas de soustraction moyenne des calques comme (0,0 , 0), la définition de l' indicateur True signifie que nous inverserons le bleu avec le rouge car OpenCV utilise BGR par contre on a des canaux dans l'image en RGB.
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),swapRB=True, crop=False)


#Visualisons ce que ressemble les 3 objets blob différents en utilisant le code suivant.
#On n'observe pas beaucoup de différence mais c'est ce qu'on va donner à l'algorithme YOLO.
#for b in blob:
#    for n,img_blob in enumerate(b):
#        cv2.imshow(str(n),img_blob)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()     
#  

 
#Ensuite, nous chargerons les 80 labels des classes dans un tableau en utilisant le fichier coco.names.



with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]
#Chargement du détecteur d'objets YOLO v3 entraîné 
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')


# Obtenir la liste des noms de toutes les couches du réseau YOLO v3 
layers_names_all = network.getLayerNames()
# Obtenir uniquement les noms de couche de sortie dont on a besoin pour l'algo YOLO
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
#affichage des couches de sorties
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']


# la probabilité minimale pour éliminer les prédictions 
probability_minimum = 0.5
#seuil lors de l'application de la technique non-maximum suppression
threshold = 0.3
#les couleurs pour représenter chaque objet détecté
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
h, w = image_BGR.shape[:2]
print('Image height={0} and width={1}'.format(h, w))  
## verification
#print(type(colours))  # <class 'numpy.ndarray'>
#print(colours.shape)  # (80, 3)
#print(colours[0])  # [172  10 127]


#Pour Transmettre l'objet blob au réseau, on utilise la fonction net.setInput (blob) , puis aux couches de sorties. Ici, tous les #objets détectés et à la sortie contiennent toutes les informations dont nous avons besoin pour extraire la position de l'objet comme #les positions en haut, à gauche, à droite, en bas, le nom de la classe.

network.setInput(blob)  
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
# le temps nécessaire pour la Pass en avant
print(' temps mit pour détection des objetc {:.5f} seconds'.format(end - start))

########### Prédiction du modèle YOLO ##############

# Preparation des listes pour les boites de delimitation, 
#les cofiances obtenues et numero de classe

bounding_boxes = []
confidences = []
class_numbers = []


# Passage par toutes les couches de sortie après le forward pass
for result in output_from_network:
    # Passage par de toutes les détections de la couche de sortie courante
    for detected_objects in result:
        #  obtenir les 80 classes de proba pour les objets détectés 
        scores = detected_objects[5:]
        # Obtenir l'indexe de la classe majoritaire
        class_current = np.argmax(scores)
        # Obtenir la valeur de proba  de cette classe
        confidence_current = scores[class_current]

        # Elimination des faibles prédictions 
        if confidence_current > probability_minimum:
            # Le format de données YOLO conserve 
            #les coordonnées du centre du rectangle de délimitation ainsi 
            #que sa largeur et sa hauteur actuelles. C'est pourquoi on dois
            #les multiplier par la largeur et la hauteur de l'image originale 
            #et obtenir ainsi les coordonnées du centre du rectangle 
            #de délimitation, sa largeur et sa hauteur pour l'image originale.
            box_current = detected_objects[0:4] * np.array([w, h, w, h])
 
            x_center, y_center, box_width, box_height = box_current
            #cv2.circle(image_BGR,(x_center, y_center),10,(0,255,0),2)
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # ajouter aux listes
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)


# un compteur pour les objets détecter
counter = 1

# Verifie s'il y a un objet détecté après le non-maximum suppression
if len(results) > 0:
    # Passage par les indexes des resultats
    for i in results.flatten():
        # le label de l'object détecté
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        # Incréménter le compteur
        counter += 1

        # Obtenir les coordonnées de la boite,
     
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # la couleur de la boite actuelle
        #conversion d'un tableau numpy à une liste
        colour_box_current = colours[class_numbers[i]].tolist()

        
        # Dessiner la boite de délimitation sur l'image originale
        cv2.rectangle(image_BGR, (x_min, y_min),(x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        # Le texte pour le label et la confiance de la boite actuelle
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Mettre ce texte sur l'image originale
        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)


# Le nombre d'objets qui ont été détecté avant et après la technique
print()
print("Le total d'object detectés =", len(bounding_boxes))
print("Nombre d'objets restants après une non maximum suppression ", counter - 1)

# affichage des objets détectés dans l'image
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Detections')

