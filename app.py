import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os

# -------------------------------
# 1. Load Model
# -------------------------------
@st.cache_resource
def load_model():
    # prefer a fine-tuned model if available
    candidate_paths = ["model_finetuned", "model"]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        st.error("⚠️ No model folder found. Please check 'model/' or 'model_finetuned/'.")
        # still attempt to load 'model' to raise a clearer error downstream
        model_path = "model"

    st.info(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path)
    return model

model = load_model()

# -------------------------------
# 2. Load Disease Data
# -------------------------------
@st.cache_data
def load_disease_data():
    with open("disease_data.json", "r") as f:
        raw = json.load(f)

    # Normalize entries so the rest of the app can rely on consistent keys.
    data = []
    descriptions = []
    for entry in raw:
        # name: prefer 'name' then 'disease_name'
        name = entry.get("name") or entry.get("disease_name") or entry.get("disease") or "Unknown"

        # description: prefer explicit 'description' else compose from common fields
        if "description" in entry and entry.get("description"):
            description = entry.get("description")
        else:
            parts = []
            for key in ("leaf_symptoms", "disease_conditions", "leaf_effects", "fruit_effects"):
                if key in entry and entry.get(key):
                    parts.append(entry.get(key))
            # fallback to the full JSON entry as a string if nothing else
            description = "\n\n".join(parts) if parts else json.dumps(entry)

        # reasoning: small summary for UI (prefer 'reasoning' if present)
        reasoning = entry.get("reasoning") or entry.get("leaf_symptoms") or "No reasoning available."

        # details: keep original entry
        details = entry

        mapped = {
            "name": name,
            "description": description,
            "reasoning": reasoning,
            "details": details,
        }

        data.append(mapped)
        descriptions.append(description)

    # create embeddings from the constructed descriptions
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    return data, embeddings

disease_data, corpus_embeddings = load_disease_data()

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Tomato Disease Predictor", page_icon="🍅", layout="wide")

st.title("🍅 Tomato Disease Prediction Assistant")
st.markdown("""
Describe the crop problems below in your own words.
The assistant will identify the most likely disease and explain why.
""")

# Input Fields
col1, col2, col3 = st.columns(3)

with col1:
    quantitative = st.text_area("📊 Quantitative Info (e.g., yield loss %)", height=120)
with col2:
    visual = st.text_area("👁️ Visual Symptoms (e.g., brown spots, yellowing leaves)", height=120)
with col3:
    weather = st.text_area("☁️ Weather/Condition Info (e.g., humid, rainy, warm)", height=120)

# Button
if st.button("🔍 Predict Disease"):
    user_input = " ".join([quantitative, visual, weather]).strip()

    if not user_input:
        st.warning("⚠️ Please describe the symptoms before predicting.")
    else:
        with st.spinner("Analyzing symptoms..."):
            query_embedding = model.encode(user_input, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            # get top-1 match safely (values and indices are length-1 tensors)
            top_values, top_indices = torch.topk(cos_scores, k=1)
            top_value = float(top_values[0].item())
            index = int(top_indices[0].item())
            matched_disease = disease_data[index]
            similarity = top_value * 100

        # Display results
        st.subheader(f"🧬 Predicted Disease: **{matched_disease['name']}**")
        st.progress(similarity / 100)
        st.write(f"**Confidence:** {similarity:.2f}%")
        st.markdown(f"**Reasoning:** {matched_disease['reasoning']}")

        with st.expander("📋 Disease Details"):
            st.json(matched_disease["details"])

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and SentenceTransformers (MiniLM).")
