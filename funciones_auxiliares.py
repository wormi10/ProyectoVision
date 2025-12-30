"""
Funciones auxiliares para extracción de características
Sistema de Clasificación de Manzanas - OpenCV
"""

import cv2
import numpy as np

# ============================================================
# DESCRIPTORES HOG (Histogram of Oriented Gradients)
# ============================================================
def calcular_hog(img_gray):
    """
    Calcula descriptores HOG de una imagen en escala de grises.
    HOG es útil para capturar información de forma y textura.
    
    Parámetros:
    - img_gray: Imagen en escala de grises
    
    Retorna:
    - Vector de características HOG normalizado
    """
    # Redimensionar si es necesario para HOG
    img_resized = cv2.resize(img_gray, (64, 64))
    
    # Configuración de HOG
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Crear descriptor HOG
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Calcular descriptores
    hog_features = hog.compute(img_resized)
    
    # Normalizar
    hog_features = hog_features.flatten()
    hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-7)
    
    return hog_features

# ============================================================
# CARACTERÍSTICAS GEOMÉTRICAS
# ============================================================
def extraer_caracteristicas_geometricas(mask):
    """
    Extrae características geométricas de un objeto segmentado:
    - Área
    - Perímetro
    - Circularidad
    Parámetros:
    - mask: Máscara binaria del objeto
    
    Retorna:
    - Vector de características geométricas
    """
    caracteristicas = []
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Si no hay contornos, devolver vector de ceros
        return np.zeros(14)
    
    # Tomar el contorno más grande (asumiendo que es la fruta)
    contour = max(contours, key=cv2.contourArea)
    
    # 1. ÁREA
    area = cv2.contourArea(contour)
    caracteristicas.append(area)
    
    # 2. PERÍMETRO
    perimeter = cv2.arcLength(contour, True)
    caracteristicas.append(perimeter)
    
    # 3. CIRCULARIDAD (4π*área/perímetro²)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0
    caracteristicas.append(circularity)
    
    return np.array(caracteristicas)
