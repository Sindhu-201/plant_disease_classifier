import streamlit as st
import torch
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plant_disease_classifier import PlantDiseaseModel, predict_image
import warnings

warnings.filterwarnings("ignore")
# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .prediction-header {
        font-size: 2rem;
        color: #2E7D32;
        margin-top: 0.5rem;
    }
    .confidence-text {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #ddd;
    }
    .disease-info-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .disease-info-subheader {
        font-size: 1.2rem;
        font-weight: bold;
        color: #43A047;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }
    .chart-container {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1.5rem;
    }
    .example-btn {
        margin: 0.2rem;
    }
    .stTable {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .upload-section {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .results-section {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Function to load model and necessary files


@st.cache_resource
def load_model_resources():
    # Load model configuration
    with open("models/model_config.json", "r") as f:
        config = json.load(f)

    # Load class names
    with open(config["class_names_path"], "r") as f:
        class_names = json.load(f)

    # Load label encoder
    with open(config["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)

    # Load image transformation
    with open(config["transform_path"], "rb") as f:
        transform = pickle.load(f)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(
        config["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, transform, label_encoder, class_names, device, config

# Function to make prediction


def predict(image_file, model, transform, label_encoder, device):
    # Save uploaded file temporarily
    with open("temp_upload.jpg", "wb") as f:
        f.write(image_file.getvalue())

    # Make prediction
    class_name, confidence, all_probs = predict_image(
        model, "temp_upload.jpg", transform, device, label_encoder
    )

    # Get top 5 predictions
    class_indices = np.argsort(all_probs)[::-1][:5]
    top_classes = [label_encoder.inverse_transform(
        [idx])[0] for idx in class_indices]

    # Format the class names for display
    formatted_top_classes = [format_class_name(
        class_name) for class_name in top_classes]
    formatted_primary_class = format_class_name(class_name)

    top_probabilities = [all_probs[idx] * 100 for idx in class_indices]

    # Remove temporary file
    os.remove("temp_upload.jpg")

    # Return both original and formatted class names
    return class_name, formatted_primary_class, confidence, top_classes, formatted_top_classes, top_probabilities

# Function to format class names for display


def format_class_name(name):
    return name.replace("_", " ").title().replace("  ", " ").replace("  ", " ")

# Function to display improved visualization of the prediction


def display_prediction(original_class, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities):
    # Main prediction header
    st.markdown(
        f"<h2 class='prediction-header'>Diagnosis: {formatted_class}</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='confidence-text'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

    # Create a container for the chart
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    # Create DataFrame for top predictions
    prediction_df = pd.DataFrame({
        "Disease": formatted_top_classes,
        "Confidence": top_probabilities
    })

    # Create improved bar chart with better styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom color palette - green gradient
    colors = plt.cm.Greens(np.linspace(0.6, 0.95, len(prediction_df)))

    bars = ax.barh(prediction_df["Disease"],
                   prediction_df["Confidence"], color=colors)

    # Improve chart appearance
    ax.set_xlabel("Confidence (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Disease", fontsize=12, fontweight='bold')
    ax.set_title("Top 5 Predictions", fontsize=16, fontweight='bold', pad=20)

    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11)

    # Add percentage labels with better formatting
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{prediction_df['Confidence'][i]:.2f}%",
                va='center', fontsize=10, fontweight='bold')

    # Set background color
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')

    # Tight layout
    plt.tight_layout()

    # Display the chart
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Create a clean table view of the predictions
    st.markdown("<h3 class='sub-header'>Detailed Predictions:</h3>",
                unsafe_allow_html=True)
    styled_df = prediction_df.style.format(
        {"Confidence": "{:.2f}%"}).background_gradient(cmap='Greens', subset=['Confidence'])
    st.table(styled_df)

# Function to display disease information


def display_disease_info(class_name, class_names):
    disease_info = {
        "Pepper__bell___Bacterial_spot": {
            "description": "Bacterial spot causes dark, water-soaked lesions on leaves and fruit, leading to defoliation and reduced yield.",
            "causes": "Caused by Xanthomonas campestris pv. vesicatoria, often spread through contaminated seeds and splashing water.",
            "treatment": "Apply copper-based bactericides and remove infected plant parts.",
            "prevention": "Use disease-free seeds, avoid overhead watering, and ensure good air circulation."
        },
        "Pepper__bell___healthy": {
            "description": "Healthy bell pepper plants exhibit firm stems, dark green leaves, and robust fruit development.",
            "causes": "Optimal growth conditions with proper nutrition, watering, and pest control.",
            "treatment": "Maintain proper plant care and monitoring to prevent diseases.",
            "prevention": "Regular fertilization, proper spacing, and pest management."
        },
        "Potato___Early_blight": {
            "description": "Early blight results in brown, concentric ring lesions on leaves, leading to defoliation and reduced tuber quality.",
            "causes": "Caused by Alternaria solani, thriving in warm, humid conditions.",
            "treatment": "Apply fungicides like chlorothalonil or mancozeb and remove infected foliage.",
            "prevention": "Rotate crops, ensure proper spacing, and use resistant potato varieties."
        },
        "Potato___Late_blight": {
            "description": "Late blight causes dark, water-soaked lesions on leaves and tubers, leading to rapid decay.",
            "causes": "Caused by Phytophthora infestans, favored by cool, moist conditions.",
            "treatment": "Use fungicides like metalaxyl and promptly remove infected plants.",
            "prevention": "Plant resistant varieties, ensure good drainage, and avoid overhead watering."
        },
        "Potato___healthy": {
            "description": "Healthy potato plants have lush green foliage, strong stems, and well-developed tubers.",
            "causes": "Proper soil preparation, watering, and pest management.",
            "treatment": "Maintain regular care and disease monitoring.",
            "prevention": "Practice crop rotation, ensure balanced fertilization, and control pests."
        },
        "Tomato_Bacterial_spot": {
            "description": "Bacterial spot causes small, dark lesions on leaves and fruit, reducing yield and quality.",
            "causes": "Caused by Xanthomonas campestris pv. vesicatoria, spread through contaminated tools and water.",
            "treatment": "Use copper-based sprays and remove infected leaves.",
            "prevention": "Avoid overhead irrigation, sanitize tools, and use disease-free seeds."
        },
        "Tomato_Early_blight": {
            "description": "Early blight leads to brown, concentric ring spots on leaves and stems, weakening the plant.",
            "causes": "Caused by Alternaria solani, thriving in warm, humid conditions.",
            "treatment": "Apply fungicides and remove affected plant parts.",
            "prevention": "Practice crop rotation, ensure proper plant spacing, and use resistant varieties."
        },
        "Tomato_Late_blight": {
            "description": "Late blight causes water-soaked lesions on leaves and fruit, leading to rapid plant decline.",
            "causes": "Caused by Phytophthora infestans, spreading in cool, wet conditions.",
            "treatment": "Use fungicides like metalaxyl and remove infected plants.",
            "prevention": "Avoid overhead watering, increase air circulation, and plant resistant varieties."
        },
        "Tomato_Leaf_Mold": {
            "description": "Leaf mold appears as yellow spots on leaves, leading to reduced photosynthesis and yield loss.",
            "causes": "Caused by Passalora fulva, thriving in high humidity.",
            "treatment": "Apply fungicides and improve air circulation.",
            "prevention": "Ensure proper spacing, prune excess foliage, and avoid overhead watering."
        },
        "Tomato_Septoria_leaf_spot": {
            "description": "Septoria leaf spot causes small, dark lesions with a yellow halo, leading to premature leaf drop.",
            "causes": "Caused by Septoria lycopersici, thriving in wet conditions.",
            "treatment": "Use fungicides and remove infected leaves.",
            "prevention": "Practice crop rotation, ensure good air circulation, and water at the base."
        },
        "Tomato_Spider_mites_Two_spotted_spider_mite": {
            "description": "Spider mites cause yellowing and stippling on leaves, leading to plant weakening.",
            "causes": "Caused by Tetranychus urticae, thriving in hot, dry conditions.",
            "treatment": "Use insecticidal soap or neem oil.",
            "prevention": "Regularly mist plants, introduce natural predators like ladybugs, and avoid drought stress."
        },
        "Tomato__Target_Spot": {
            "description": "Target spot appears as dark, concentric lesions on leaves and stems, weakening the plant.",
            "causes": "Caused by Corynespora cassiicola, spreading in humid environments.",
            "treatment": "Apply fungicides and remove infected plant parts.",
            "prevention": "Ensure proper spacing, prune excess foliage, and maintain dry foliage."
        },
        "Tomato__Tomato_YellowLeaf__Curl_Virus": {
            "description": "This viral disease causes yellowing and curling of leaves, leading to stunted growth.",
            "causes": "Spread by whiteflies.",
            "treatment": "No direct cure; manage whitefly populations with insecticides and resistant varieties.",
            "prevention": "Use reflective mulches, introduce natural predators, and remove infected plants."
        },
        "Tomato__Tomato_mosaic_virus": {
            "description": "Mosaic virus causes mottled, distorted leaves and reduced fruit yield.",
            "causes": "Spread through infected seeds, tools, and human handling.",
            "treatment": "No cure; remove infected plants and sanitize tools.",
            "prevention": "Use virus-free seeds, wash hands before handling plants, and control insect vectors."
        },
        "Tomato_healthy": {
            "description": "Healthy tomato plants show vibrant green leaves, strong stems, and normal fruit development.",
            "causes": "Proper care, adequate watering, good sunlight exposure, and regular fertilization.",
            "treatment": "Continue regular care practices to maintain plant health.",
            "prevention": "Regular monitoring, balanced nutrition, appropriate watering, and good air circulation."
        }
    }

    if class_name in disease_info:
        st.markdown("<div class='section-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<h3 class='disease-info-header'>Disease Information</h3>", unsafe_allow_html=True)

        info = disease_info[class_name]

        # Create two columns for the disease information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<h4 class='disease-info-subheader'>Description</h4>", unsafe_allow_html=True)
            st.write(info["description"])

            st.markdown(
                "<h4 class='disease-info-subheader'>Causes</h4>", unsafe_allow_html=True)
            st.write(info["causes"])

        with col2:
            st.markdown(
                "<h4 class='disease-info-subheader'>Treatment</h4>", unsafe_allow_html=True)
            st.write(info["treatment"])

            st.markdown(
                "<h4 class='disease-info-subheader'>Prevention</h4>", unsafe_allow_html=True)
            st.write(info["prevention"])
    else:
        st.info("Detailed information for this specific plant condition is not available. Please consult with an agricultural expert.")


def main():
    try:
        model, transform, label_encoder, class_names, device, config = load_model_resources()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False

    # App title and header
    st.markdown("<h1 class='main-header'>🌿 Plant Disease Classifier</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='info-text'>This application uses deep learning to diagnose diseases in plant leaves. Simply upload an image of a plant leaf, and the model will predict if it's healthy or identify the disease.</p>", unsafe_allow_html=True)

    # Create two columns for upload and image display
    col1, col2 = st.columns([1, 1])

    # Store uploaded file in a session state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Upload Image</h3>",
                    unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

        # Example images section with better layout
        st.markdown("<h3 class='sub-header'>Or try an example:</h3>",
                    unsafe_allow_html=True)

        # Use columns for better button layout
        example_col1, example_col2, example_col3 = st.columns(3)

        if example_col1.button("Healthy Tomato", key="example1", use_container_width=True):
            st.session_state.uploaded_file = "images/examples/tomato_healthy.jpg"

        if example_col2.button("Potato Late Blight", key="example2", use_container_width=True):
            st.session_state.uploaded_file = "images/examples/Potato_Late_blight.jpeg"

        if example_col3.button("Pepper Bacterial Spot", key="example3", use_container_width=True):
            st.session_state.uploaded_file = "images/examples/Pepper_bell_Bacterial_spot.jpeg"

        # Display available classes in an expander
        with st.expander("Available Plant Diseases for Classification"):
            # Format class names for display
            formatted_classes = [name.replace("_", " ").replace(
                "__", " ").replace("___", " ").title() for name in class_names]

            # Create a clean table for the class names
            classes_df = pd.DataFrame(
                {"Available Diseases": formatted_classes})
            st.table(classes_df)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='results-section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Image Preview</h3>",
                    unsafe_allow_html=True)

        # Display the uploaded image with better styling
        if st.session_state.uploaded_file is not None:
            # Handle both string paths (example images) and uploaded file objects
            if isinstance(st.session_state.uploaded_file, str):
                image = Image.open(st.session_state.uploaded_file)
                st.image(image, caption="Selected Image",
                         use_container_width=True)

                # Process the file for prediction
                with open(st.session_state.uploaded_file, "rb") as f:
                    file_content = f.read()
                    uploaded_file_obj = type('obj', (object,), {
                        'getvalue': lambda: file_content
                    })

                # Make prediction
                st.session_state.prediction_results = predict(
                    uploaded_file_obj, model, transform, label_encoder, device
                )
                st.session_state.image_processed = True

            else:
                # Handle normal uploaded file
                image = Image.open(st.session_state.uploaded_file)
                st.image(image, caption="Uploaded Image",
                         use_container_width=True)

                # Make prediction
                st.session_state.prediction_results = predict(
                    st.session_state.uploaded_file, model, transform, label_encoder, device
                )
                st.session_state.image_processed = True
        else:
            # Placeholder when no image is uploaded
            st.info(
                "Please upload an image or select an example to see the preview and diagnosis.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction section that spans the full width
    if st.session_state.image_processed and st.session_state.prediction_results is not None:
        st.markdown("<div class='section-divider'></div>",
                    unsafe_allow_html=True)

        # Unpack prediction results
        class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities = st.session_state.prediction_results

        # Display prediction with improved visualization
        display_prediction(class_name, formatted_class, confidence,
                           top_classes, formatted_top_classes, top_probabilities)

        # Display disease information
        display_disease_info(class_name, class_names)

    # Show model information in sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #2E7D32;'>About the Model</h2>",
                    unsafe_allow_html=True)
        st.write(
            "This application uses a Convolutional Neural Network (CNN) to classify plant diseases from leaf images.")

        st.markdown(
            "<h3 style='color: #43A047; margin-top: 20px;'>Model Architecture:</h3>", unsafe_allow_html=True)
        if model_loaded:
            st.write(
                "- **Model Type:** CNN with 5 convolutional blocks with advanced architectures")
            st.write(f"- **Number of Classes:** {len(class_names)}")
            st.write("- **Test Accuracy:** 96.5%")

        st.markdown(
            "<h3 style='color: #43A047; margin-top: 20px;'>Dataset Information:</h3>", unsafe_allow_html=True)
        st.write("- Trained on the PlantVillage dataset")
        st.write("- Contains 15 classes of plant diseases and healthy plants")
        st.write("- Classes include diseases in tomatoes, potatoes, and peppers")

        st.markdown(
            "<h3 style='color: #43A047; margin-top: 20px;'>Usage Instructions:</h3>", unsafe_allow_html=True)
        st.write("1. Upload an image of a plant leaf")
        st.write("2. View the diagnosis and recommended actions")
        st.write("3. Check detailed information about the disease")

        



if __name__ == "__main__":
    main()
