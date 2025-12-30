"""
Script para clasificar nuevas imágenes de manzanas
usando el modelo entrenado
"""

import cv2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from funciones_auxiliares import *

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH = 'modelo_manzanas.pkl'
SCALER_PATH = 'scaler.pkl'

# ============================================================
# CARGAR MODELO
# ============================================================
def cargar_modelo():
    """
    Carga el modelo y el scaler guardados
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            modelo = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return modelo, scaler
    except FileNotFoundError:
        print("❌ Error: No se encontró el modelo entrenado.")
        print("Ejecuta primero 'main.py' para entrenar el modelo.")
        sys.exit(1)

# ============================================================
# PREPROCESAMIENTO (igual que en main.py)
# ============================================================
def preprocesar_imagen(img):
    """Preprocesa la imagen"""
    img_suavizada = cv2.GaussianBlur(img, (5, 5), 1.5)
    lab = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_mejorada = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_mejorada

def segmentar_fruta(img):
    """Segmenta la fruta del fondo"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    img_segmentada = cv2.bitwise_and(img, img, mask=mask)
    return img_segmentada, mask

def extraer_caracteristicas(img):
    """Extrae características de la imagen"""
    img_prep = preprocesar_imagen(img)
    img_seg, mask = segmentar_fruta(img_prep)
    
    hsv = cv2.cvtColor(img_seg, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [32], [0, 256])
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
    hog_features = calcular_hog(gray)
    caracteristicas_geo = extraer_caracteristicas_geometricas(mask)
    
    caracteristicas = np.concatenate([hist_h, hist_s, hist_v, hog_features, caracteristicas_geo])
    return caracteristicas, img_seg, mask

# ============================================================
# CLASIFICACIÓN
# ============================================================
def clasificar_imagen(ruta_imagen, modelo, scaler, nombres_clases, visualizar=True):
    """
    Clasifica una nueva imagen de manzana
    
    Parámetros:
    - ruta_imagen: Path a la imagen
    - modelo: Modelo SVM entrenado
    - scaler: Normalizador
    - nombres_clases: Lista con nombres de las clases
    - visualizar: Si True, muestra la imagen con la predicción
    
    Retorna:
    - clase_predicha: Nombre de la clase
    - probabilidades: Array con probabilidades de cada clase
    """
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"❌ Error: No se pudo cargar la imagen '{ruta_imagen}'")
        return None, None
    
    # Extraer características
    caracteristicas, img_seg, mask = extraer_caracteristicas(img)
    caracteristicas = caracteristicas.reshape(1, -1)
    
    # Normalizar
    caracteristicas_scaled = scaler.transform(caracteristicas)
    
    # Predecir
    prediccion = modelo.predict(caracteristicas_scaled)[0]
    probabilidades = modelo.predict_proba(caracteristicas_scaled)[0]
    
    clase_predicha = nombres_clases[prediccion]
    
    # Visualizar resultado
    if visualizar:
        visualizar_resultado(img, img_seg, mask, clase_predicha, probabilidades, nombres_clases)
    
    return clase_predicha, probabilidades

# ============================================================
# VISUALIZACIÓN
# ============================================================
def visualizar_resultado(img_original, img_segmentada, mask, clase_predicha, 
                        probabilidades, nombres_clases):
    """
    Visualiza el resultado de la clasificación
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Imagen original
    ax1 = plt.subplot(1, 3, 1)
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title('Imagen Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Imagen segmentada
    ax2 = plt.subplot(1, 3, 2)
    img_seg_rgb = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_seg_rgb)
    ax2.set_title('Segmentación', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Gráfico de probabilidades
    ax3 = plt.subplot(1, 3, 3)
    colores = ['#FF6B6B' if i == np.argmax(probabilidades) else '#4ECDC4' 
               for i in range(len(nombres_clases))]
    barras = ax3.barh(nombres_clases, probabilidades * 100, color=colores)
    ax3.set_xlabel('Probabilidad (%)', fontsize=12)
    ax3.set_title(f'Predicción: {clase_predicha}', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 100])
    
    # Añadir valores en las barras
    for i, (barra, prob) in enumerate(zip(barras, probabilidades)):
        width = barra.get_width()
        ax3.text(width + 2, barra.get_y() + barra.get_height()/2, 
                f'{prob*100:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# CLASIFICACIÓN EN LOTE
# ============================================================
def clasificar_carpeta(ruta_carpeta, modelo, scaler, nombres_clases):
    """
    Clasifica todas las imágenes en una carpeta
    
    Parámetros:
    - ruta_carpeta: Path a la carpeta con imágenes
    - modelo: Modelo entrenado
    - scaler: Normalizador
    - nombres_clases: Lista con nombres de clases
    """
    import os
    
    archivos = [f for f in os.listdir(ruta_carpeta) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(archivos) == 0:
        print(f"❌ No se encontraron imágenes en '{ruta_carpeta}'")
        return
    
    print(f"\nClasificando {len(archivos)} imágenes de '{ruta_carpeta}'...\n")
    print("="*60)
    
    resultados = []
    
    for archivo in archivos:
        ruta_img = os.path.join(ruta_carpeta, archivo)
        clase, probs = clasificar_imagen(ruta_img, modelo, scaler, nombres_clases, visualizar=False)
        
        if clase is not None:
            prob_max = np.max(probs) * 100
            resultados.append({
                'archivo': archivo,
                'clase': clase,
                'confianza': prob_max
            })
            print(f"✓ {archivo:30s} → {clase:20s} ({prob_max:.1f}%)")
    
    print("="*60)
    print(f"\nTotal clasificadas: {len(resultados)}/{len(archivos)} imágenes")
    
    return resultados

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def main():
    """
    Función principal para clasificar imágenes
    """
    print("\n" + "="*60)
    print("CLASIFICADOR DE MANZANAS")
    print("="*60 + "\n")
    
    # Cargar modelo
    print("Cargando modelo entrenado...")
    modelo, scaler = cargar_modelo()
    
    # Lista de clases (debe coincidir con las del entrenamiento)
    # NOTA: Ajusta estos nombres según tus clases reales
    nombres_clases = ['Apple_Braeburn', 'Apple_Granny_Smith', 'Apple_Golden', 'Apple_Red']
    print(f"✓ Modelo cargado. Clases: {nombres_clases}\n")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python clasificar_imagen.py <ruta_imagen>")
        print("  python clasificar_imagen.py <ruta_carpeta> --carpeta")
        print("\nEjemplo:")
        print("  python clasificar_imagen.py manzana.jpg")
        print("  python clasificar_imagen.py imagenes_test/ --carpeta")
        return
    
    ruta = sys.argv[1]
    
    # Clasificar carpeta completa
    if len(sys.argv) > 2 and sys.argv[2] == '--carpeta':
        clasificar_carpeta(ruta, modelo, scaler, nombres_clases)
    # Clasificar imagen individual
    else:
        clase, probs = clasificar_imagen(ruta, modelo, scaler, nombres_clases)
        if clase is not None:
            print(f"\n{'='*60}")
            print(f"Resultado: {clase}")
            print(f"Confianza: {np.max(probs)*100:.2f}%")
            print(f"{'='*60}\n")

if __name__ == "__main__":
    main()