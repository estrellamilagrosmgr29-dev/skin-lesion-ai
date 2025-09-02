import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image, ImageOps
import cv2

# ===============================
# Configuración de la app
# ===============================
st.set_page_config(page_title="DermAI", layout="centered")
st.title("🌟 DermAI: Tu piel bajo cuidado inteligente mediante visión por computadora 🌟")


# Cargar modelo CNN entrenado
@st.cache_resource
def load_cnn_model():
    try:
        # Asegúrate de que la ruta al modelo sea correcta
        model = load_model("/modelo_lunares.h5")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_cnn_model()

# Clases
clases = ["Benigno", "Maligno", "Sospechoso"]

# ===============================
# Función predicción CNN
# ===============================
def predict_cnn(model, pil_img):
    try:
        image = pil_img.convert("RGB")
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        confianza = np.max(prediction)

        return clases[class_idx], confianza
    except Exception as e:
        return "Error CNN", 0.0

# ===============================
# Función análisis ABCDE
# ===============================
def evaluar_abcde(pil_img):
    image = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    # A - Asimetría (relación perímetro/área)
    A = (perimeter ** 2) / (4 * np.pi * area + 1e-5)

    # B - Bordes (irregularidad)
    B = len(c)

    # C - Colores (número de colores únicos aproximado)
    C = len(np.unique(image.reshape(-1, 3), axis=0))

    # D - Diámetro (radio aproximado en px)
    (x,y,w,h) = cv2.boundingRect(c)
    D = max(w, h)

    return {"A": A, "B": B, "C": C, "D": D}

# ===============================
# Función de predicción principal
# ===============================
def predecir_imagen(pil_img):
    if model is None:
        return "Error: Modelo no cargado.", 0.0, {"A": -1, "B": -1, "C": -1, "D": -1}, "Error de análisis", ""

    # CNN
    clase_pred_cnn, confianza_cnn = predict_cnn(model, pil_img)


    # ABCDE
    abcde_results = evaluar_abcde(pil_img)

    # Mensaje final basado en CNN y ABCDE
    mensaje_final = "Análisis en proceso..."
    if clase_pred_cnn == "Error CNN":
        mensaje_final = "❌ Error en la predicción CNN. No se pudo realizar una evaluación completa."
    # Ajustar los umbrales según sea necesario
    elif clase_pred_cnn == "Maligno" or abcde_results["A"] > 0.1 or abcde_results["B"] > 50 or abcde_results["C"] > 10 or abcde_results["D"] > 6:
        mensaje_final = "⚠️ Posible lesión maligna. Debes acudir al dermatólogo lo antes posible."
    elif clase_pred_cnn == "Sospechoso":
        mensaje_final = "⚠️ Lesión sospechosa. Vigílala y consulta a un dermatólogo."
    else:
        mensaje_final = "✅ Lesión benigna. No parece preocupante, pero mantén control periódico."

    recomendaciones = """
### Recomendaciones Generales de Protección Solar:
- Evita el sol directo entre las 10 a.m. y 4 p.m.
- Busca la sombra.
- Usa ropa protectora, sombrero y gafas.
- Aplica protector solar SPF 30+ cada 2 horas.
- Reaplica después de nadar o sudar.
- Mantente hidratado.
"""
