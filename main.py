"""
Sistema de Detección y Clasificación de Manzanas con OpenCV
Trabajo Práctico - Visión por Computador
Autores: Samuel Vera García, Pablo Rodríguez Ballesteros
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from funciones_auxiliares import *

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_PATH = 'dataset'  # Ruta a tu carpeta dataset
MODEL_PATH = 'modelo_manzanas.pkl'
SCALER_PATH = 'scaler.pkl'
TEST_SIZE = 0.2  # 80% entrenamiento, 20% prueba
RANDOM_STATE = 123

# ============================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# ============================================================
def cargar_dataset(dataset_path):
    """
    Carga imágenes desde carpetas organizadas por clase.
    Estructura esperada:
    dataset/
        ├── Apple_Braeburn/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── Apple_Granny_Smith/
        └── Apple_Red/
    """
    print("="*60)
    print("CARGANDO DATASET")
    print("="*60)
    
    imagenes = []
    etiquetas = []
    nombres_clases = []
    
    # Listar todas las carpetas (clases)
    clases = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    clases.sort()
    
    print(f"Clases encontradas: {clases}")
    print()
    
    for idx, clase in enumerate(clases):
        clase_path = os.path.join(dataset_path, clase)
        archivos = [f for f in os.listdir(clase_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Clase '{clase}': {len(archivos)} imágenes")
        
        for archivo in archivos:
            img_path = os.path.join(clase_path, archivo)
            img = cv2.imread(img_path)
            
            if img is not None:
                imagenes.append(img)
                etiquetas.append(idx)
                if clase not in nombres_clases:
                    nombres_clases.append(clase)
    
    print()
    print(f"Total de imágenes cargadas: {len(imagenes)}")
    print(f"Total de clases: {len(nombres_clases)}")
    
    return np.array(imagenes), np.array(etiquetas), nombres_clases

# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================


def preprocesar_imagen(img):
    """
    Preprocesado compatible con histogramas HSV:
    - Suavizado Gaussiano (reduce ruido)
    - Normalización ligera
    NO cambia el espacio de color
    """

    # Suavizado para reducir ruido
    img_suavizada = cv2.GaussianBlur(img, (5, 5), 1.5)

    # Normalización opcional (mantiene colores)
    img_suavizada = cv2.normalize(
        img_suavizada, None, 0, 255, cv2.NORM_MINMAX
    )

    return img_suavizada


# ============================================================
# 3. SEGMENTACIÓN
# ============================================================
def segmentar_fruta(img):
    """
    Segmenta la fruta del fondo usando:
    - Conversión a HSV
    - Umbralización
    - Operaciones morfológicas
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Máscara para detectar objetos no blancos (fondo típico)
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Aplicar máscara a la imagen original
    img_segmentada = cv2.bitwise_and(img, img, mask=mask)
    
    return img_segmentada, mask

# ============================================================
# 4. EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def extraer_caracteristicas(img):
    """
    Extrae un vector de características combinando:
    1. Histogramas de color en HSV (más robusto que RGB)
    2. Descriptores HOG para textura/forma
    3. Características geométricas de contornos
    """
    # Preprocesar
    img_prep = preprocesar_imagen(img)
    
    # Segmentar
    img_seg, mask = segmentar_fruta(img_prep)
    
    # 1. HISTOGRAMAS DE COLOR HSV
    hsv = cv2.cvtColor(img_seg, cv2.COLOR_BGR2HSV)
    
    # Histograma de Hue (matiz - color)
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    
    # Histograma de Saturation (saturación)
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    
    # Histograma de Value (valor/brillo)
    hist_v = cv2.calcHist([hsv], [2], mask, [32], [0, 256])
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # 2. DESCRIPTORES HOG
    gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
    hog_features = calcular_hog(gray)
    
    # 3. CARACTERÍSTICAS GEOMÉTRICAS
    caracteristicas_geo = extraer_caracteristicas_geometricas(mask)
    
    # Combinar todas las características
    caracteristicas = np.concatenate([
        hist_h,
        hist_s, 
        hist_v,
        hog_features,
        caracteristicas_geo
    ])
    
    return caracteristicas

# ============================================================
# 5. ENTRENAMIENTO DEL CLASIFICADOR
# ============================================================
def entrenar_modelo(X_train, y_train):
    """
    Entrena un clasificador SVM con kernel RBF
    """
    print("="*60)
    print("ENTRENANDO MODELO SVM")
    print("="*60)
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entrenar SVM con kernel RBF
    modelo = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE)
    modelo.fit(X_train_scaled, y_train)
    
    print("Modelo entrenado correctamente")
    
    return modelo, scaler

# ============================================================
# 6. EVALUACIÓN DEL MODELO
# ============================================================
def evaluar_modelo(modelo, scaler, X_test, y_test, nombres_clases):
    """
    Evalúa el modelo y muestra métricas y visualizaciones
    """
    print("="*60)
    print("EVALUANDO MODELO")
    print("="*60)
    
    # Normalizar datos de prueba
    X_test_scaled = scaler.transform(X_test)
    
    # Predicciones
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Precisión (Accuracy): {accuracy*100:.2f}%\n")
    
    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=nombres_clases))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=300)
    print("Matriz de confusión guardada en 'matriz_confusion.png'")
    plt.show()
    
    return y_pred, accuracy
def visualizar_preprocesado_y_segmentacion(imagenes, nombres_clases=None, etiquetas=None):

    """
    Muestra un ejemplo aleatorio del dataset:
    - Imagen original
    - Imagen preprocesada
    - Máscara de segmentación
    - Imagen segmentada
    """

    # Elegir imagen aleatoria
    idx = np.random.randint(0, len(imagenes))
    img = imagenes[idx]

    # Preprocesado
    img_prep = preprocesar_imagen(img)

    # Segmentación
    img_seg, mask = segmentar_fruta(img_prep)

    # Convertir a RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_prep_rgb = cv2.cvtColor(img_prep, cv2.COLOR_BGR2RGB)
    img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)

    # Título con clase si existe
    titulo = "Ejemplo aleatorio"
    if etiquetas is not None and nombres_clases is not None:
        titulo += f" - Clase: {nombres_clases[etiquetas[idx]]}"

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Imagen original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_prep_rgb)
    axes[0, 1].set_title("Imagen preprocesada")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title("Máscara de segmentación")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_seg_rgb)
    axes[1, 1].set_title("Imagen segmentada")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================
def visualizar_ejemplos(imagenes_test, y_test, y_pred, nombres_clases, n_ejemplos=9):
    """
    Muestra ejemplos de predicciones
    """
    print("\nGenerando visualización de ejemplos...")
    
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(imagenes_test), min(n_ejemplos, len(imagenes_test)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = cv2.cvtColor(imagenes_test[idx], cv2.COLOR_BGR2RGB)
        real = nombres_clases[y_test[idx]]
        pred = nombres_clases[y_pred[idx]]
        correcto = y_test[idx] == y_pred[idx]
        
        axes[i].imshow(img)
        color = 'green' if correcto else 'red'
        axes[i].set_title(f'Real: {real}\nPredicho: {pred}', color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ejemplos_clasificacion.png', dpi=300)
    print(" Ejemplos guardados en 'ejemplos_clasificacion.png'")
    plt.show()

# ============================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================
def main():
    """
    Pipeline completo de clasificación
    """
    print("\n" + "="*60)
    print("SISTEMA DE CLASIFICACIÓN DE MANZANAS")
    print("Visión por Computador con OpenCV")
    print("="*60 + "\n")
    
    # 1. Cargar dataset
    imagenes, etiquetas, nombres_clases = cargar_dataset(DATASET_PATH)
    
    if len(imagenes) == 0:
        print(" Error: No se encontraron imágenes en el dataset")
        return
    visualizar_preprocesado_y_segmentacion(imagenes, nombres_clases, etiquetas)
    # 2. Dividir en train y test
    print("\n" + "="*60)
    print("DIVIDIENDO DATASET")
    print("="*60)
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        imagenes, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )
    print(f"Conjunto de entrenamiento: {len(X_train_img)} imágenes")
    print(f"Conjunto de prueba: {len(X_test_img)} imágenes")
    
    # 3. Extraer características
    print("\n" + "="*60)
    print("EXTRAYENDO CARACTERÍSTICAS")
    print("="*60)
    print("Procesando imágenes de entrenamiento...")
    X_train = np.array([extraer_caracteristicas(img) for img in X_train_img])
    print(f" Características de entrenamiento: {X_train.shape}")
   
    print("Procesando imágenes de prueba...")
    X_test = np.array([extraer_caracteristicas(img) for img in X_test_img])
    print(f" Características de prueba: {X_test.shape}")
    
    # 4. Entrenar modelo
    modelo, scaler = entrenar_modelo(X_train, y_train)
    
    # 5. Guardar modelo
    print("\nGuardando modelo...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(modelo, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Modelo guardado en '{MODEL_PATH}'")
    print(f"✓ Scaler guardado en '{SCALER_PATH}'")
    
    # 6. Evaluar modelo
    y_pred, accuracy = evaluar_modelo(modelo, scaler, X_test, y_test, nombres_clases)
    
    # 7. Visualizar resultados
    visualizar_ejemplos(X_test_img, y_test, y_pred, nombres_clases)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    print(f"✓ Precisión final: {accuracy*100:.2f}%")
    print(f"✓ Modelo entrenado y guardado")
    print(f"✓ Visualizaciones generadas")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()