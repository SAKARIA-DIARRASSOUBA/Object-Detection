import numpy as np
import cv2
import time
 
## Détection d'objets sur une vidéo avec la bibliothèque OpenCV du deep learning 
##
## Algorithm:
## Lire une video en entrée --> Charger le réseau  YOLO v3 -->
## --> Lire les images dans une boucle --> Obtenir un blob par image -->
## --> Implementer la fonction Forward Pass --> Obtenir les boites de délimitations -->
## --> Non-maximum Suppression --> Dessiner les boites de délimitations avec les Labels -->
## --> Écriture des images traitées
##
## Resultat:
## Nouveau fichier de video avec les objets détectés les boites de délimitations et les Labels
#
#
#"""
#lecture d'ume video 
#
#video= cv2.VideoCapture('videos/traffic-cars-and-people.mp4')
#
#while(video.isOpened()):
#    ret, frame = video.read()
#    # utiliser la couleur noir et blanc dans toute la video
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    
#    # Pour garder l'image de la video
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(5) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()
#
#Fin de lecture de la video
#"""
#
#"""
#Lecture de video en entrée
#"""
#
##  'VideoCapture' pemet de lire une vidéo à partir d'un fichier
##video = cv2.VideoCapture('videos/traffic-cars.mp4')
#video = cv2.VideoCapture('videos/Famille-Thevenin.mp4')
##  redacteur pour les images traitées
#writer = None
#
##  variables pour les dimensions spatiales des images 
#h, w = None, None
#
#"""
#End of:
#Reading input video
#"""
#    
#"""
#Start of:
#Loading YOLO v3 network
#"""    
#with open('yolo-coco-data/coco.names') as f:
#    labels = [line.strip() for line in f]
##Chargement du détecteur d'objets YOLO v3 entraîné 
#network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
#                                     'yolo-coco-data/yolov3.weights')
#
#
## Obtenir la liste des noms de toutes les couches du réseau YOLO v3 
#layers_names_all = network.getLayerNames()
## Obtenir uniquement les noms de couche de sortie dont on a besoin pour l'algo YOLO
#layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
##affichage des couches de sorties
#print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']
#
#
## la probabilité minimale pour éliminer les prédictions 
#probability_minimum = 0.5
##seuil lors de l'application de la technique non-maximum suppression
#threshold = 0.3
##les couleurs pour représenter chaque objet détecté
#colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
#
#
#"""
#End of:
#Loading YOLO v3 network
#"""
#
#"""
#Start of:
#Lecture das images dans la boucle
#"""
#
##Definir une variable pour compter les images et à la fin on aura le total des images traitées 
#f = 0
#
##Definir une variable temps  et à la fin on aura le temps mis pour toutes  images traitées 
#t = 0
#
#while True:
#    # Capturer image par image
#    ret, frame = video.read()
#
#    # Si l'image n'est pas récupérer , à la fin de la video on arrete la boucle
#    if not ret:
#        break
#
#    # Obtenir les dimensions spatiales de l'image
#    if w is None or h is None:
#        # Découpage du tuple uniquement des deux premiers éléments
#        h, w = frame.shape[:2]
#
#    """
#    Start of:
#    Getting blob from current frame
#    """
#
#    #L'entrée du réseau est ce que l'on appelle un objet blob . La fonction transforme l'image 
#    #en un blob via cv2.dnn.blobFromImage. Il a les paramètres suivants:
#    #l' image à transformer
#    #le facteur d' échelle (1/255 pour mettre à l'échelle les valeurs de pixel à [0..1])
#    #la taille , ici une image carrée  416×416 
#    #la valeur moyenne (par défaut = 0)
#    #l'option swapBR = True (car OpenCV utilise BGR)
#    #la définition de l' indicateur True signifie que nous inverserons le bleu avec le rouge car OpenCV utilise BGR par contre on a des canaux dans l'image en RGB.
#    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
#                                 
#
#    """
#    End of:
#    Getting blob from current frame
#    """
#
#    """
#    Start of:
#    Implementing Forward pass
#    """
#
#    # Implementer le  forward pass avec notre  blob et seulement à travers  les couchs de sorties
#    # en calculant au meme moment le temps nécessaire pour le forward pass
#    network.setInput(blob)  
#    start = time.time()
#    output_from_network = network.forward(layers_names_output)
#    end = time.time()
#
#    # Incrementer les compteurs pour les images et le temps
#    f += 1
#    t += end - start
#    print('Image numéro {0} a mis {1:.5f} seconds'.format(f, end - start))
#
#    """
#    End of:
#    Implementing Forward pass
#    """
#
#    """
#    Start of:
#    Getting bounding boxes
#    """
#    # Preparation des listes pour les boites de delimitation, 
#    #les cofiances obtenues et numero de classe
#
#    bounding_boxes = []
#    confidences = []
#    class_numbers = []
#
#
#    # Passage par toutes les couches de sortie après le forward pass
#    for result in output_from_network:
#        # Passage par de toutes les détections de la couche de sortie courante
#        for detected_objects in result:
#            #  obtenir les 80 classes de proba pour les objets détectés 
#            scores = detected_objects[5:]
#            # Obtenir l'indexe de la classe majoritaire
#            class_current = np.argmax(scores)
#            # Obtenir la valeur de proba  de cette classe
#            confidence_current = scores[class_current]
#
#            # Elimination des faibles prédictions 
#            if confidence_current > probability_minimum:
#                # Le format de données YOLO conserve 
#                #les coordonnées du centre du rectangle de délimitation ainsi 
#                #que sa largeur et sa hauteur actuelles. C'est pourquoi on dois
#                #les multiplier par la largeur et la hauteur de l'image originale 
#                #et obtenir ainsi les coordonnées du centre du rectangle 
#                #de délimitation, sa largeur et sa hauteur pour l'image originale.
#                box_current = detected_objects[0:4] * np.array([w, h, w, h])
#
#                x_center, y_center, box_width, box_height = box_current
#                #cv2.circle(image_BGR,(x_center, y_center),10,(0,255,0),2)
#                x_min = int(x_center - (box_width / 2))
#                y_min = int(y_center - (box_height / 2))
#
#                # ajouter aux listes
#                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
#                confidences.append(float(confidence_current))
#                class_numbers.append(class_current)
#
#    
#    """
#    End of:
#    Getting bounding boxes
#    
#    """
#    """
#    Start of:
#    Non-maximum suppression
#    """
#
#    # Implementer le non-maximum suppression pour les boites de délimitation
#    # Avec cette technique, on exclut certaines boites de confiances faibles 
#    #au détriment d'une boite de la région avec une grande confiance  
#    
#    # C'est nécessaire de s'assurer que les types données de la boite soient 'int' 
#    # et les types données des confiances soient 'float' pour plus d'infos voir le lien suivant
#    # https://github.com/opencv/opencv/issues/12789
#    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
#                               
#
#    """
#    End of:
#    Non-maximum suppression
#    """
#
#    
#    """
#    Start of:
#    Drawing bounding boxes and labels
#    """
#
#    # Verifie s'il y a un objet détecté après le non-maximum suppression
#    if len(results) > 0:
#        # Passage par les indexes des resultats
#        for i in results.flatten():
#            # Obtenir les coordonnées de la boite,
#            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
#            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
#
#            # la couleur de la boite actuelle
#            #conversion d'un tableau numpy à une liste
#            colour_box_current = colours[class_numbers[i]].tolist()
#
#
#            # Dessiner la boite de délimitation sur l'image originale
#            cv2.rectangle(frame, (x_min, y_min),(x_min + box_width, y_min + box_height),
#                          colour_box_current, 2)
#
#            # Le texte pour le label et la confiance de la boite actuelle
#            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
#                                                   confidences[i])
#
#            # Mettre ce texte sur l'image originale
#            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
#                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
#
#
#
#    """
#    End of:
#    Drawing bounding boxes and labels
#    """
#
#    """
#    Start of:
#    Writing processed frame into the file
#    """
#
#    # Initialiser le redacteur
#    
#    if writer is None:
#        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
#        # Ecriture de l'image traitée actuelle dans le fichier video
#        writer = cv2.VideoWriter('videos/result-traffic-cars.mp4', fourcc, 30,
#                                 (frame.shape[1], frame.shape[0]), True)
#
#    writer.write(frame)
#
#    """
#    End of:
#    Writing processed frame into the file
#    """
#
#"""
#End of:
#Reading frames in the loop
#"""
#
#
## Affichage du resultat final
#print()
#print("Nombres total de l'images", f)
#print('Temps mis {:.5f} seconds'.format(t))
#print("Image Par Seconde :", round((f / t), 1))
#
#video.release()
#writer.release()
#
#
#
#""""
#Start of 
#Lecture de la vidéo resultante
#
#
#Lors de l'affichage du cadre, utilisez le temps approprié pour cv2.waitKey(). 
#S'il est trop petit, la vidéo sera très rapide et si elle est trop élevée, 
#la vidéo sera lente (c'est ainsi que vous pouvez afficher des vidéos au ralenti). 
#25 millisecondes seront OK dans des cas normaux.
#"""
video= cv2.VideoCapture('videos/result-Famille.mp4')

while(video.isOpened()):
    ret, frame = video.read()
    # utiliser la couleur noir et blanc dans toute la video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pour garder l'image de la video
    cv2.imshow('frame',frame)
    if cv2.waitKey(35) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

""""
End of 
Lecture de la vidéo resultante
"""       