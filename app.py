import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo treinado
model = load_model('modelo_iris_scan.h5')

# Classes (ordem deve bater com o treinamento)
classes = ['iris-setosa', 'iris-versicolor', 'iris-virginica']

# Título do app
st.title('IrisScan 🌸 - Classificador de Flores Iris')

# Upload da imagem
uploaded_file = st.file_uploader("Faça upload de uma imagem de flor Iris...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Imagem enviada.', use_column_width=True)

    # Pré-processar imagem
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer predição
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    # Mostrar o resultado
    st.subheader('Predição:')
    st.write(f'A flor é provavelmente: **{predicted_class}** 🌸')