
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt # Importar matplotlib


# ===============================
# Cargar modelo
# ===============================
# Verificar si el modelo ya está cargado para evitar recargarlo
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("modelo_lunares.h5") # Asumiendo que el modelo está en el mismo directorio o en una ruta accesible
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_trained_model()
clases = ["benignos", "malignos", "sospechosos"]
img_size = (128,128)

# ===============================
# Función para evaluar ABCDE
# ===============================
def evaluar_abcde(pil_img):
    img = np.array(pil_img.convert('RGB')) # Asegurar 3 canales
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Umbral adaptativo o manual con manejo de errores
    try:
        # Intentar umbral adaptativo primero si la imagen es lo suficientemente grande
        if gray.shape[0] > 50 and gray.shape[1] > 50: # Umbral mínimo razonable
             thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else: # Si la imagen es pequeña, usar umbral manual simple
             _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    except Exception as e:
        st.warning(f"Advertencia en umbralización: {e}. Usando umbral manual simple.")
        try:
             _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        except Exception as e_manual:
             st.error(f"Error grave en umbralización manual: {e_manual}")
             return {"asimetria": -1, "bordes": -1, "colores": -1, "diametro": -1, "error": "Umbralización fallida"}


    # A: Asimetría (comparar lados)
    h, w = thresh.shape
    asimetria = -1.0 # Valor por defecto en caso de error

    # Verificar si el ancho es par y mayor que cero antes de dividir
    if w > 0 and w % 2 == 0:
        try:
            left = thresh[:, :w//2]
            right = cv2.flip(thresh[:, w//2:], 1)
            # Asegurarse de que las dimensiones coincidan antes de absdiff
            if left.shape == right.shape:
                 asimetria = np.sum(cv2.absdiff(left, right)) / (h*w)
            else:
                 st.warning(f"Advertencia: Dimensiones no coinciden ({left.shape} vs {right.shape}) para asimetría. W={w}")
                 asimetria = -1.0
        except Exception as e:
            st.warning(f"Advertencia calculando asimetría: {e}. W={w}")
            asimetria = -1.0
    else:
        st.warning(f"Advertencia: Ancho impar o cero para asimetría. W={w}")
        asimetria = -1.0


    # B: Bordes (detección de bordes con Canny)
    irregularidad = -1.0 # Valor por defecto en caso de error
    try:
        bordes = cv2.Canny(thresh, 100, 200)
        irregularidad = np.sum(bordes) / (h*w) if h*w > 0 else -1.0 # Evitar división por cero
    except Exception as e:
        st.warning(f"Advertencia calculando bordes: {e}")
        irregularidad = -1.0


    # C: Colores
    colores = -1 # Valor por defecto en caso de error
    try:
        # Asegurarse de que la imagen sea a color para contar colores únicos en 3 canales
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convertir a un tipo de datos que permita np.unique con axis=0 si es necesario, o simplemente a bytes para ser eficiente
            # Para evitar problemas con tipos de datos, a veces es más simple aplanar y usar set
            pixels = img.reshape(-1, 3)
            # Convertir cada pixel (array de 3 elementos) a una tupla para que sea hasheable y funcione con set
            unique_pixels = set(tuple(p) for p in pixels)
            colores = len(unique_pixels)
        else:
             st.warning("Advertencia: Imagen no tiene 3 canales para cálculo de colores.")
             # Si es escala de grises, contar niveles de gris únicos
             if len(img.shape) == 2:
                 colores = len(np.unique(img))
             else:
                 colores = -1 # No se puede calcular
    except Exception as e:
        st.warning(f"Advertencia calculando colores: {e}")
        colores = -1


    # D: Diámetro (radio aproximado en px)
    diametro = 0 # Valor por defecto
    try:
        # Asegurarse de que thresh no esté vacío o inválido
        if thresh is not None and np.sum(thresh) > 0:
             contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             if contornos:
                 # Encontrar el contorno más grande, asegurándose de que tenga un área válida
                 largest_contour = max(contornos, key=cv2.contourArea)
                 if cv2.contourArea(largest_contour) > 0:
                     (x,y,wc,hc) = cv2.boundingRect(largest_contour)
                     diametro = max(wc, hc)
                 else:
                     st.warning("Advertencia: Contorno encontrado pero con área cero.")
             else:
                 st.warning("Advertencia: No se encontraron contornos.")
        else:
             st.warning("Advertencia: La imagen binaria (thresh) está vacía o inválida para encontrar contornos.")
    except Exception as e:
        st.warning(f"Advertencia calculando diámetro: {e}")
        diametro = 0


    return {
        "asimetria": asimetria,
        "bordes": irregularidad,
        "colores": colores,
        "diametro": diametro,
        "error": None # Indicar que el análisis se intentó
    }

# ===============================
# Función de predicción principal
# ===============================
def predecir_imagen(pil_img):
    if model is None:
        return "Error: Modelo no cargado.", 0.0, {"asimetria": -1, "bordes": -1, "colores": -1, "diametro": -1, "error": "Modelo no disponible"}

    # CNN
    try:
        img_resized = pil_img.resize(img_size)
        x = image.img_to_array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)
        clase_pred = clases[np.argmax(pred)]
        confianza = np.max(pred)
    except Exception as e:
        st.error(f"Error en la predicción CNN: {e}")
        clase_pred = "Error CNN"
        confianza = 0.0


    # ABCDE - Incluir manejo de errores al llamar a evaluar_abcde
    abcde = evaluar_abcde(pil_img)


    return clase_pred, confianza, abcde

# ===============================
# Interfaz de Streamlit
# ===============================
st.title("DermaAI: Tu piel bajo cuidado inteligente mediante vision por computadora")
st.write("Sube una imagen de una lesión cutánea para obtener una predicción.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("Analizar imagen"):
        st.spinner("Analizando...")

        # Realizar la predicción
        clase_pred, confianza, abcde_results = predecir_imagen(image)

        st.subheader("Resultados del Análisis:")

        # Mostrar resultado CNN
        if clase_pred != "Error CNN":
             st.write(f"**Predicción (CNN):** {clase_pred} ({confianza*100:.2f}% de confianza)")
        else:
             st.error(f"**Predicción (CNN):** {clase_pred}. Consulta los logs para más detalles.")


        # Mostrar resultados ABCDE
        st.subheader("Análisis ABCDE:")
        if abcde_results.get("error") is None:
            st.write(f"- **A (Asimetría):** {abcde_results['asimetria']:.4f}")
            st.write(f"- **B (Bordes):** {abcde_results['bordes']:.4f}")
            st.write(f"- **C (Colores):** {abcde_results['colores']}")
            st.write(f"- **D (Diámetro):** {abcde_results['diametro']} px")
        else:
            st.warning(f"Análisis ABCDE no pudo completarse: {abcde_results['error']}")
            st.write("No se pudieron calcular los criterios ABCDE para esta imagen.")


        # Mensaje final basado en CNN y ABCDE (con manejo de posibles errores en ABCDE)
        st.subheader("Evaluación Combinada:")
        mensaje_final = "Evaluación en proceso..." # Mensaje por defecto si hay errores
        recomendaciones = """
### Recomendaciones Generales de Protección Solar:
- Evita el sol directo entre las 10 a.m. y 4 p.m.
- Busca la sombra.
- Usa ropa protectora, sombrero y gafas.
- Aplica protector solar SPF 30+ cada 2 horas.
- Reaplica después de nadar o sudar.
- Mantente hidratado.
"""
        abcde_ok = abcde_results.get("error") is None and -1 not in abcde_results.values()

        if clase_pred == "Error CNN":
             mensaje_final = "❌ Error en la predicción CNN. No se pudo realizar una evaluación completa."
        elif not abcde_ok:
             mensaje_final = f"⚠️ Predicción CNN: **{clase_pred}**. Error en el análisis ABCDE. Considera consultar a un especialista si tienes dudas."
        elif clase_pred == "malignos" or abcde_results["asimetria"] > 0.1 or abcde_results["bordes"] > 0.05 or abcde_results["colores"] > 50 or abcde_results["diametro"] > 100:
            mensaje_final = "**🔴 Posible lesión maligna.** **Debes acudir al dermatólogo lo antes posible.**"
        elif clase_pred == "sospechosos":
            mensaje_final = "**🟠 Lesión sospechosa.** Vigílala y consulta a un dermatólogo."
        else:
            mensaje_final = "**🟢 Lesión benigna.** No parece preocupante, pero mantén control periódico."

        st.markdown(mensaje_final)
        st.markdown(recomendaciones)

        # Mostrar imagen con resultado (opcional, si quieres superponer algo)
        # Puedes añadir código aquí para dibujar contornos, etc. si deseas visualizarlos
        # Por ahora, simplemente mostramos la imagen original arriba
