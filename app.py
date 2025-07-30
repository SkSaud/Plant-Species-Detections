import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import time

# Load trained TensorFlow model
MODEL_PATH = "C:/plant species detection/plant_species_model.h5"
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:/plant species detection/plant_species_model.h5")

model = load_model()

# Preprocess Image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Plant Class Labels
CLASS_LABELS = [
    "ZZ Plant (Zamioculcas zamiifolia)", "Yucca", "Venus Flytrap", "Tulip", "Tradescantia",
    "Snake plant (Sanseviera)", "Schefflera", "Sago Palm (Cycas revoluta)", "Rubber Plant (Ficus elastica)",
    "Rattlesnake Plant (Calathea lancifolia)", "Prayer Plant (Maranta leuconeura)", "Pothos (Ivy arum)", 
    "Ponytail Palm (Beaucarnea recurvata)", "Polka Dot Plant (Hypoestes phyllostachya)", "Poinsettia (Euphorbia pulcherrima)", 
    "Peace lily", "Parlor Palm (Chamaedorea elegans)", "Orchid", "Monstera Deliciosa (Monstera deliciosa)", 
    "Money Tree (Pachira aquatica)", "Lily of the valley (Convallaria majalis)", "Lilium (Hemerocallis)", 
    "Kalanchoe", "Jade plant (Crassula ovata)", "Iron Cross begonia (Begonia masoniana)", 
    "Hyacinth (Hyacinthus orientalis)", "English Ivy (Hedera helix)", "Elephant Ear (Alocasia spp.)", 
    "Dumb Cane (Dieffenbachia spp.)", "Dracaena", "Daffodils (Narcissus spp.)", "Ctenanthe", "Chrysanthemum", 
    "Christmas Cactus (Schlumbergera bridgesii)", "Chinese evergreen (Aglaonema)", "Chinese Money Plant (Pilea peperomioides)", 
    "Cast Iron Plant (Aspidistra elatior)", "Calathea", "Boston Fern (Nephrolepis exaltata)", 
    "Birds Nest Fern (Asplenium nidus)", "Bird of Paradise (Strelitzia reginae)", "Begonia (Begonia spp.)", 
    "Asparagus Fern (Asparagus setaceus)", "Areca Palm (Dypsis lutescens)", "Anthurium (Anthurium andraeanum)", 
    "Aloe Vera", "African Violet (Saintpaulia ionantha)"
]
PLANT_INFO = {
    "ZZ Plant (Zamioculcas zamiifolia)": "🌿 Drought-tolerant, thrives in low light, requires infrequent watering. Prefers well-draining soil and temperatures of 60-75°F.",
    "Yucca": "🌞 Loves bright, indirect light. Drought-resistant, only needs occasional watering. Prefers sandy, well-draining soil and warm temperatures.",
    "Venus Flytrap": "🪰 Carnivorous plant that traps insects. Needs bright light, distilled water, and high humidity. Grows best in nutrient-poor, acidic soil.",
    "Tulip": "🌷 A seasonal flowering plant that needs full sun and well-draining soil. Requires moderate watering and cooler temperatures in winter for proper blooming.",
    "Tradescantia": "💜 Fast-growing with striking purple-green leaves. Prefers bright, indirect light, moist soil, and high humidity.",
    "Snake Plant (Sansevieria)": "🌱 Air-purifying, tolerates neglect and low light. Needs occasional watering and thrives in sandy, well-draining soil.",
    "Schefflera": "🌳 Also known as the 'Umbrella Tree.' Prefers indirect sunlight, moderate watering, and well-draining soil.",
    "Sago Palm (Cycas revoluta)": "🌴 Slow-growing with a tropical appearance. Requires bright, indirect light, well-draining soil, and occasional watering.",
    "Rubber Plant (Ficus elastica)": "🌿 Large, glossy leaves. Thrives in indirect light, prefers moist soil, and benefits from occasional leaf wiping to remove dust.",
    "Rattlesnake Plant (Calathea lancifolia)": "🍃 Unique wavy leaves with vibrant patterns. Needs bright, indirect light, high humidity, and moist soil.",
    "Prayer Plant (Maranta leuconeura)": "🙏 Leaves move up at night. Prefers medium to low light, moist soil, and high humidity.",
    "Pothos (Ivy arum)": "🌿 Hardy vine that purifies air. Tolerates low light, requires occasional watering, and thrives in almost any soil type.",
    "Ponytail Palm (Beaucarnea recurvata)": "🌴 Drought-resistant, stores water in its bulbous trunk. Needs bright light and occasional watering.",
    "Polka Dot Plant (Hypoestes phyllostachya)": "🎨 Colorful, spotted leaves. Prefers bright, indirect light and consistently moist soil.",
    "Poinsettia (Euphorbia pulcherrima)": "🎄 Bright red holiday plant. Requires bright, indirect light and moderate watering.",
    "Peace Lily": "🌼 Elegant white flowers. Prefers shade, moist soil, and high humidity. Air-purifying and easy to care for.",
    "Parlor Palm (Chamaedorea elegans)": "🌴 Low-light tolerant palm. Requires moderate watering and well-draining soil.",
    "Orchid": "🌸 Needs bright, indirect light, high humidity, and well-draining bark-based soil. Requires careful watering to prevent root rot.",
    "Monstera Deliciosa (Monstera deliciosa)": "🍃 Large split leaves, loves bright, indirect light and moist, well-draining soil.",
    "Money Tree (Pachira aquatica)": "💰 Symbol of prosperity. Thrives in bright, indirect light and needs consistent moisture.",
    "Lily of the Valley (Convallaria majalis)": "🌿 Fragrant, bell-shaped white flowers. Prefers partial shade, moist soil, and cool temperatures.",
    "Lilium (Hemerocallis)": "🌺 Stunning blooms, requires full sun, well-draining soil, and moderate watering.",
    "Kalanchoe": "🌵 Succulent with colorful flowers. Loves bright light, occasional watering, and well-draining soil.",
    "Jade Plant (Crassula ovata)": "💰 Succulent that thrives in bright light, requires minimal watering, and is known as a good luck plant.",
    "Iron Cross Begonia (Begonia masoniana)": "🍂 Unique foliage with dark cross patterns. Prefers shade, high humidity, and moist soil.",
    "Hyacinth (Hyacinthus orientalis)": "🌸 Fragrant flowers, needs full sun, well-draining soil, and moderate watering.",
    "English Ivy (Hedera helix)": "🌿 Fast-growing vine, air-purifying, prefers bright indirect light and occasional watering.",
    "Elephant Ear (Alocasia spp.)": "🌿 Large tropical leaves, requires high humidity, indirect light, and consistently moist soil.",
    "Dumb Cane (Dieffenbachia spp.)": "🌱 Toxic if ingested, thrives in indirect light, needs well-draining soil and moderate watering.",
    "Dracaena": "🌴 Hardy plant, prefers low to medium light, well-draining soil, and occasional watering.",
    "Daffodils (Narcissus spp.)": "🌼 Spring flowers, need full sun, well-draining soil, and moderate watering.",
    "Ctenanthe": "🍃 Ornamental foliage, needs bright indirect light, high humidity, and moist soil.",
    "Chrysanthemum": "🌼 Bright, long-lasting flowers. Requires full sun and frequent watering.",
    "Christmas Cactus (Schlumbergera bridgesii)": "🎄 Blooms in winter, requires indirect light, well-draining soil, and moderate watering.",
    "Chinese Evergreen (Aglaonema)": "🌱 Easy-care plant, tolerates low light, needs moderate watering, and humid conditions.",
    "Chinese Money Plant (Pilea peperomioides)": "💰 Round coin-shaped leaves, needs bright indirect light, well-draining soil, and occasional watering.",
    "Cast Iron Plant (Aspidistra elatior)": "🛡️ Extremely hardy, tolerates low light, minimal watering, and neglect.",
    "Calathea": "🍃 Ornamental plant with moving leaves. Needs indirect light, high humidity, and consistently moist soil.",
    "Boston Fern (Nephrolepis exaltata)": "🌿 Loves humidity and bright, indirect light. Requires frequent watering.",
    "Birds Nest Fern (Asplenium nidus)": "🐦 Wavy leaves, thrives in warm, humid environments with indirect light.",
    "Bird of Paradise (Strelitzia reginae)": "🌺 Exotic tropical plant with large leaves and striking flowers. Needs bright light.",
    "Begonia (Begonia spp.)": "🌸 Beautiful flowers with unique foliage. Prefers high humidity and indirect light.",
    "Asparagus Fern (Asparagus setaceus)": "🌿 Delicate feathery fronds, prefers bright indirect light and high humidity.",
    "Areca Palm (Dypsis lutescens)": "🌴 Air-purifying palm, thrives in bright, indirect light and requires moderate watering.",
    "Anthurium (Anthurium andraeanum)": "❤️ Heart-shaped red flowers. Prefers warmth, humidity, and moist soil.",
    "Aloe Vera": "🌵 Medicinal succulent, thrives in bright light, requires minimal watering, and well-draining soil.",
    "African Violet (Saintpaulia ionantha)": "🌸 Small flowers with fuzzy leaves, blooms year-round, prefers bright indirect light and moist soil."
}

# 🌱 **Main UI**
st.title("🌿 AI Plant Species Detector")
st.write("Upload an image of a house plant, and the AI will identify the species!")

st.sidebar.title("📘 Plant Species Info")
selected_plant = st.sidebar.selectbox("Select a plant species:", PLANT_INFO)

# ✅ Show information about the selected plant
st.sidebar.write(PLANT_INFO[selected_plant])


# **📷 Image Upload**
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")


    # **🎭 Real-time Animation for Prediction**
    status = st.empty()  # Placeholder for status update
    status.text("🔍 Analyzing Image...")
    time.sleep(2)

    # **📊 AI Prediction**
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    species = CLASS_LABELS[predicted_class]

    # 🎯 **Prediction Confidence**
    confidence_scores = predictions[0] * 100  # Convert to percentage
    top_5_indices = np.argsort(confidence_scores)[-5:][::-1]  # Top 5 predictions
    top_5_species = [CLASS_LABELS[i] for i in top_5_indices]
    top_5_confidence = [confidence_scores[i] for i in top_5_indices]

    # **✅ Show Prediction**
    status.success(f"🌿 **Predicted Species:** {species} ({confidence_scores[predicted_class]:.2f}%)")

    # **📊 Confidence Score Bar Chart**
    fig = px.bar(
        x=top_5_species,
        y=top_5_confidence,
        labels={'x': 'Plant Species', 'y': 'Confidence (%)'},
        title="Top 5 Predicted Species with Confidence Scores",
        color=top_5_species,
    )
    st.plotly_chart(fig)

    # **📊 Pie Chart: Prediction Confidence**
    fig_pie = px.pie(
        names=top_5_species,
        values=top_5_confidence,
        title="AI Confidence Distribution",
        hole=0.4,
        color=top_5_species
    )
    st.plotly_chart(fig_pie)

    # **📋 Table: Prediction Breakdown**
    df = pd.DataFrame({
        "Plant Species": top_5_species,
        "Confidence (%)": top_5_confidence
    })
    st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"))

    # **📘 Show Plant Information**
    st.subheader(f"📖 About {species}")
    if species in CLASS_LABELS:
        st.write(f"🌱 *{species}* is a commonly known house plant. It thrives in suitable light conditions and requires proper care.")

    # 🎭 **Dynamic Progress Bar**
    st.subheader("🌿 AI Model Confidence Progress")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)