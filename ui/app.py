import streamlit as st
import requests
import os

API = os.environ.get("API_URL", "https://sumative-plant-disease-classifier.onrender.com/")

DISEASE_INFO = {
    "Tomato_Bacterial_Spot": (
        "Bacterial Spot",
        "Caused by Xanthomonas bacteria — small, dark, water-soaked spots on leaves and fruit, common in warm, wet weather.",
    ),
    "Tomato_Early_Blight": (
        "Early Blight",
        "Caused by the fungus Alternaria solani — dark spots with concentric \"target\" rings on older, lower leaves.",
    ),
    "Tomato_Healthy": (
        "Healthy",
        "No signs of disease detected on this leaf.",
    ),
    "Tomato_Late_Blight": (
        "Late Blight",
        "Caused by Phytophthora infestans — dark, greasy-looking blotches that spread quickly in cool, wet weather.",
    ),
    "Tomato_Leaf_Mold": (
        "Leaf Mold",
        "Caused by a fungus (Passalora fulva) — yellow spots on top of leaves with olive-green fuzzy mold underneath, common in humid conditions.",
    ),
    "Tomato_Septoria_Leaf_Spot": (
        "Septoria Leaf Spot",
        "Caused by the fungus Septoria lycopersici — many small circular spots with dark borders and light gray centers.",
    ),
    "Tomato_Spider_Mites_Two_Spotted_Spider_Mite": (
        "Spider Mites (Two-Spotted)",
        "A pest infestation, not a fungus or bacteria — causes stippled, yellowing leaves and fine webbing.",
    ),
    "Tomato_Target_Spot": (
        "Target Spot",
        "Caused by the fungus Corynespora cassiicola — brown lesions with concentric rings on leaves, stems, and fruit.",
    ),
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": (
        "Yellow Leaf Curl Virus",
        "A viral disease spread by whiteflies — causes upward-curling, yellowing leaves and stunted growth.",
    ),
    "Tomato_Tomato_mosaic_virus": (
        "Mosaic Virus",
        "A viral disease causing mottled light/dark green patterns and leaf distortion, easily spread by contact or tools.",
    ),
    "Not_Tomato_Leaf": (
        "Not a Tomato Leaf",
        "This image doesn't appear to show a tomato leaf at all.",
    ),
}


def describe_prediction(label: str, confidence: float) -> str:
    display_name, description = DISEASE_INFO.get(label, (label.replace("_", " "), ""))
    text = f"**Diagnosis: {display_name}** ({confidence:.2%} confidence)"
    if description:
        text += f"\n\n{description}"
    return text


st.set_page_config(page_title="Plant Disease Classsifier", layout="wide")
st.title("🌿 Plant Disease Classifier — ML Ops Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Model Uptime", "Data Visualizations", "Predict", "Upload & Retrain"
    ])

with tab1:
    st.header("API Health")
    try:
        r = requests.get(f"{API}/health", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Status", r.get("status", "unknown"))
        col2.metric("Uptime (seconds)", r.get("uptime_seconds", 0))
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


with tab2:
    st.subheader("Dataset Visualizations")
    st.write("Generate from the training notbook")
    for img_name , caption in [
        ("class_distribution.png", "Class distribution — checks for class imbalance."),
        ("sample_images.png", "Sample images per class — visual separability check."),
        ("color_histogram.png", "Average color profile per class — diseased vs healthy signal."),
        ("confusion_matrix.png", "Confusion matrix on the held-out test set."),

    ]: 
        path = f"notebook/{img_name}"
        if os.path.exists(path):
            st.image(path, caption=caption, width="stretch")
        else:
            st.info(f"Missing : {path} - Generate it in the notebook first . ")


with tab3:
    st.subheader("Predict on a single leaf image")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], key="predict_file_uploader")
    if uploaded_file and st.button("Predict.....", key="predict_button"):
        resp = requests.post(
             f"{API}/predict",
             files={"file" : (uploaded_file.name, uploaded_file.getvalue())},

        ).json()
        if "Error" in resp or "error" in resp:
            st.error(resp)
        elif resp["predicted_label"] == "Uncertain":
            st.image(uploaded_file, width=300)
            st.warning(f"{resp['message']} ({resp['confidence']:.2%} confidence)")
        else:
            st.image(uploaded_file, width=300)
            st.success(describe_prediction(resp["predicted_label"], resp["confidence"]))

with tab4:
    st.subheader("Bulk upload new training data")
    class_name = st.text_input("Class name for these images (e.g. Tomato_Healthy)", key="class_name_input")
    bulk_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk_file_uploader"
    )
    if bulk_files and class_name and st.button("Upload Images", key="upload_images_button"):
        files_payload = [("files", (f.name, f.getvalue())) for f in bulk_files]
        resp = requests.post(
            f"{API}/upload",
            data={"class_name": class_name},
            files=files_payload,
        ).json()
        st.success(resp)

    st.divider()
    st.subheader("Trigger Retraining")
    st.write("Retrains the model on all data currently in `data/train/` (original + uploaded).")
    if st.button("Retrain Model Now", key="retrain_button"):
        with st.spinner("Retraining in progress — this may take a while..."):
            resp = requests.post(f"{API}/train", timeout=1800).json()
        st.success(resp)

