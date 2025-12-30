import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm
ruta_base = "dataset"  # cambia esto por tu ruta

for carpeta_actual, subcarpetas, archivos in os.walk(ruta_base):
    for archivo in archivos:
        # Filtrar solo imágenes
        if archivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            
            ruta_imagen = os.path.join(carpeta_actual, archivo)

            # Cargar imagen
            imagen = cv2.imread(ruta_imagen)

            if imagen is None:
                print("⚠ No se pudo cargar:", ruta_imagen)
                continue

            print("📷 Imagen cargada:", ruta_imagen)

            # === Aquí procesas la imagen ===
            # ejemplo: convertir a gris
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            # ejemplo: mostrar la imagen
            # cv2.imshow("img", gris)
            # cv2.waitKey(1)

# cv2.destroyAllWindows()
            print("Procesada:", ruta_imagen) 
            # =================================

            print("📁 Carpeta actual:", carpeta_actual
                  , "| Subcarpetas:", subcarpetas
                  , "| Archivo:", archivo)
            print("--------------------------------------------------"

# Función para extraer características HOG