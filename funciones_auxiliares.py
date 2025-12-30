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
    - Relación de aspecto
    - Extent (relación área/área del bounding box)
    - Solidez (relación área/área del convex hull)
    - Momentos de Hu
    
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
    
    # 4. BOUNDING BOX Y RELACIÓN DE ASPECTO
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    caracteristicas.append(aspect_ratio)
    
    # 5. EXTENT (área / área del bounding box)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    caracteristicas.append(extent)
    
    # 6. CONVEX HULL Y SOLIDEZ
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    caracteristicas.append(solidity)
    
    # 7. MOMENTOS DE HU (invariantes a escala, rotación y traslación)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Los momentos de Hu pueden ser muy pequeños, aplicar log
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    caracteristicas.extend(hu_moments)
    
    return np.array(caracteristicas)

# ============================================================
# DESCRIPTORES ADICIONALES (OPCIONAL)
# ============================================================
def calcular_sift_features(img_gray, max_features=50):
    """
    Calcula descriptores SIFT (Scale-Invariant Feature Transform).
    NOTA: SIFT puede no estar disponible en todas las versiones de OpenCV.
    
    Parámetros:
    - img_gray: Imagen en escala de grises
    - max_features: Número máximo de keypoints a extraer
    
    Retorna:
    - Vector de características SIFT promediado
    """
    try:
        # Crear detector SIFT
        sift = cv2.SIFT_create(nfeatures=max_features)
        
        # Detectar keypoints y descriptores
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        
        if descriptors is not None and len(descriptors) > 0:
            # Promediar descriptores para obtener un vector fijo
            feature_vector = np.mean(descriptors, axis=0)
            return feature_vector
        else:
            return np.zeros(128)  # SIFT genera descriptores de 128 dimensiones
    except:
        # Si SIFT no está disponible, devolver vector de ceros
        return np.zeros(128)

def calcular_orb_features(img_gray, max_features=50):
    """
    Calcula descriptores ORB (Oriented FAST and Rotated BRIEF).
    ORB es una alternativa libre a SIFT.
    
    Parámetros:
    - img_gray: Imagen en escala de grises
    - max_features: Número máximo de keypoints a extraer
    
    Retorna:
    - Vector de características ORB promediado
    """
    # Crear detector ORB
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # Detectar keypoints y descriptores
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    if descriptors is not None and len(descriptors) > 0:
        # Convertir a float y promediar
        descriptors_float = descriptors.astype(np.float32)
        feature_vector = np.mean(descriptors_float, axis=0)
        return feature_vector
    else:
        return np.zeros(32)  # ORB genera descriptores de 32 dimensiones

# ============================================================
# ANÁLISIS DE TEXTURA CON LBP
# ============================================================
def calcular_lbp(img_gray, radius=1, n_points=8):
    """
    Calcula el patrón binario local (LBP) para análisis de textura.
    
    Parámetros:
    - img_gray: Imagen en escala de grises
    - radius: Radio del patrón circular
    - n_points: Número de puntos de muestreo
    
    Retorna:
    - Histograma LBP normalizado
    """
    # Implementación simplificada de LBP
    lbp = np.zeros_like(img_gray, dtype=np.uint8)
    
    for i in range(radius, img_gray.shape[0] - radius):
        for j in range(radius, img_gray.shape[1] - radius):
            center = img_gray[i, j]
            binary_code = 0
            
            # Comparar píxeles vecinos
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                
                if img_gray[x, y] >= center:
                    binary_code |= (1 << k)
            
            lbp[i, j] = binary_code
    
    # Calcular histograma
    hist = cv2.calcHist([lbp], [0], None, [2**n_points], [0, 2**n_points])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

# ============================================================
# ANÁLISIS DE COLOR AVANZADO
# ============================================================
def calcular_estadisticas_color(img, mask=None):
    """
    Calcula estadísticas de color: media y desviación estándar
    por cada canal HSV.
    
    Parámetros:
    - img: Imagen BGR
    - mask: Máscara opcional para análisis
    
    Retorna:
    - Vector con [mean_h, std_h, mean_s, std_s, mean_v, std_v]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    estadisticas = []
    for i in range(3):
        canal = hsv[:, :, i]
        if mask is not None:
            valores = canal[mask > 0]
        else:
            valores = canal.flatten()
        
        media = np.mean(valores)
        std = np.std(valores)
        estadisticas.extend([media, std])
    
    return np.array(estadisticas)

# ============================================================
# VISUALIZACIÓN DE PROCESAMIENTO
# ============================================================
def visualizar_procesamiento(img, titulo="Procesamiento"):
    """
    Muestra el resultado del procesamiento de una imagen.
    Útil para debugging.
    
    Parámetros:
    - img: Imagen a visualizar
    - titulo: Título de la ventana
    """
    cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualizar_comparacion(img_original, img_procesada):
    """
    Muestra una comparación lado a lado de imagen original y procesada.
    
    Parámetros:
    - img_original: Imagen original
    - img_procesada: Imagen después del procesamiento
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convertir BGR a RGB para matplotlib
    axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Procesada')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()