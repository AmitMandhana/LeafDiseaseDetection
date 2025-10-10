import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
import shap
import numpy as np

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
# SHAP Setup with Custom Wrapper
# -------------------------------
@st.cache_resource
def setup_shap_explainer():
    """Create a SHAP explainer with a custom prediction wrapper"""
    def predict_disease_probability(texts):
        """Wrapper function that returns disease probabilities for SHAP"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            query_emb = model.encode(text, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, corpus_embeddings)[0]
            # Convert to probabilities using softmax
            probs = torch.nn.functional.softmax(scores, dim=0).cpu().numpy()
            results.append(probs)
        
        return np.array(results)
    
    # Use Partition explainer with a simple callable
    return predict_disease_probability

predict_fn = setup_shap_explainer()

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
    user_input = " ".join([quantitative.strip(), visual.strip(), weather.strip()])

    if not user_input.strip():
        st.warning("⚠️ Please describe the symptoms before predicting.")
    else:
        with st.spinner("Analyzing symptoms..."):
            query_embedding = model.encode(user_input, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
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

        # Explainability: SHAP + Simple keyword analysis
        with st.spinner("Generating explanation..."):
            st.subheader("🔍 Explanation of Prediction")
            
            # Simple word-based importance by comparing with matched disease description
            matched_desc = matched_disease['description'].lower()
            input_words = user_input.lower().split()
            
            # Find overlapping words/phrases
            important_words = []
            for word in input_words:
                if len(word) > 3 and word in matched_desc:  # Filter short words
                    important_words.append(word)
            
            if important_words:
                st.write("**Key matching symptoms:**")
                st.write(", ".join(important_words))
            else:
                st.write("No direct symptom matches found, prediction based on semantic similarity.")
            
            # Show symptom breakdown by field
            st.write("\n**Your input breakdown:**")
            if quantitative.strip():
                st.write(f"- 📊 Quantitative: {quantitative.strip()}")
            if visual.strip():
                st.write(f"- 👁️ Visual: {visual.strip()}")
            if weather.strip():
                st.write(f"- ☁️ Weather: {weather.strip()}")
            
            # SHAP word-level importance
            try:
                st.write("\n**SHAP Analysis - Word Importance:**")
                with st.spinner("Computing SHAP values..."):
                    # Split input into words for token-level analysis
                    words = user_input.split()
                    if len(words) > 1:
                        # Create a custom explainer for this specific input
                        explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text(tokenizer=r"\W+"))
                        shap_values = explainer([user_input])
                        
                        # Display text plot
                        st.write("Words highlighted by importance:")
                        shap.plots.text(shap_values[0, :, index], display=False)
                        st.pyplot(bbox_inches='tight', dpi=150, pad_inches=0.1)
                    else:
                        st.info("Input too short for word-level SHAP analysis")
            except Exception as e:
                st.warning(f"SHAP analysis unavailable: {str(e)[:100]}")
                st.info("Using simple keyword matching instead.")

        with st.expander("📋 Disease Details"):
            st.json(matched_disease["details"])

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and SentenceTransformers (MiniLM).")


## Disambiguation helper
st.markdown("---")
st.header("🧭 Disambiguate between two diseases")
st.markdown("If two classes are confused, pick them below and the assistant will ask targeted questions to disambiguate.")

# build name list for selection
names = [d['name'] for d in disease_data]
col_a, col_b = st.columns(2)
with col_a:
    dis1 = st.selectbox("Select disease A", options=names, key='dis1')
with col_b:
    dis2 = st.selectbox("Select disease B", options=names, key='dis2')

if dis1 and dis2 and dis1 != dis2:
    if st.button("Start disambiguation"):
        # initialize session state for disambiguation
        st.session_state['dis_q_index'] = 0
        st.session_state['dis_scores'] = {dis1: 0, dis2: 0}
        st.session_state['dis_qs'] = []

    # helper to create distinguishing questions
    def make_questions(d1, d2, max_q=6):
        details1 = next((x['details'] for x in disease_data if x['name'] == d1), {})
        details2 = next((x['details'] for x in disease_data if x['name'] == d2), {})
        fields = ["leaf_symptoms", "fruit_effects", "disease_conditions", "leaf_effects", "plant_growth_effects"]
        qs = []
        seen = set()
        for field in fields:
            t1 = details1.get(field, "") or ""
            t2 = details2.get(field, "") or ""
            sents1 = [s.strip() for s in t1.split('.') if s.strip()]
            sents2 = [s.strip() for s in t2.split('.') if s.strip()]
            # unique sentences for d1
            for s in sents1:
                key = (s.lower(), field, d1)
                if key in seen:
                    continue
                if len(s) < 10:
                    continue
                if s.lower() not in t2.lower():
                    qtxt = f"Do you observe: '{s}' ?"
                    qs.append({"q": qtxt, "target": d1, "evidence": s, "field": field})
                    seen.add(key)
                    if len(qs) >= max_q:
                        return qs
            # unique sentences for d2
            for s in sents2:
                key = (s.lower(), field, d2)
                if key in seen:
                    continue
                if len(s) < 10:
                    continue
                if s.lower() not in t1.lower():
                    qtxt = f"Do you observe: '{s}' ?"
                    qs.append({"q": qtxt, "target": d2, "evidence": s, "field": field})
                    seen.add(key)
                    if len(qs) >= max_q:
                        return qs
        # fallback: compare short keywords
        if not qs:
            # take first informative fragments
            frag1 = (sents1[0] if sents1 else details1.get('leaf_symptoms','') )
            frag2 = (sents2[0] if sents2 else details2.get('leaf_symptoms','') )
            if frag1:
                qs.append({"q": f"Do you observe: '{frag1}' ?", "target": d1, "evidence": frag1, "field": 'fallback'})
            if frag2:
                qs.append({"q": f"Do you observe: '{frag2}' ?", "target": d2, "evidence": frag2, "field": 'fallback'})
        return qs

    # ensure session state exists
    if 'dis_qs' not in st.session_state:
        st.session_state['dis_qs'] = []
    if 'dis_q_index' not in st.session_state:
        st.session_state['dis_q_index'] = 0
    if 'dis_scores' not in st.session_state:
        st.session_state['dis_scores'] = {dis1: 0, dis2: 0}

    # start generating questions if empty
    if not st.session_state['dis_qs']:
        st.session_state['dis_qs'] = make_questions(dis1, dis2, max_q=6)
        st.session_state['dis_q_index'] = 0
        st.session_state['dis_scores'] = {dis1: 0, dis2: 0}

    qs = st.session_state['dis_qs']
    idx = st.session_state['dis_q_index']

    if qs:
        st.markdown(f"**Question {idx+1} of {len(qs)}**")
        qobj = qs[idx]
        ans = st.radio(qobj['q'], options=["Yes", "No", "Not sure"], key=f"ans_{idx}")

        if st.button("Submit answer"):
            # update scores
            target = qobj['target']
            other = dis2 if target == dis1 else dis1
            if ans == "Yes":
                st.session_state['dis_scores'][target] += 1
            elif ans == "No":
                st.session_state['dis_scores'][target] -= 1
            # advance
            st.session_state['dis_q_index'] = min(idx+1, len(qs)-1)
            # if we haven't reached end move to next, else show results
            if st.session_state['dis_q_index'] == idx:
                # end reached; do nothing
                pass
            else:
                st.experimental_rerun()

        if st.button("Skip question"):
            st.session_state['dis_q_index'] = min(idx+1, len(qs)-1)
            st.experimental_rerun()

        # show running scores
        st.write("Current scores:")
        st.write(st.session_state['dis_scores'])

        # if at last question, allow final decision
        if idx == len(qs)-1:
            if st.button("Finish and show result"):
                scores = st.session_state['dis_scores']
                d_a, d_b = dis1, dis2
                s_a, s_b = scores.get(d_a,0), scores.get(d_b,0)
                if s_a > s_b:
                    winner = d_a
                elif s_b > s_a:
                    winner = d_b
                else:
                    winner = None

                if winner:
                    st.success(f"🔎 Based on your answers, the most likely disease is: {winner}")
                else:
                    st.info("🔎 The answers are inconclusive between the two diseases.")

                # reasoning: show matched evidence
                st.markdown("**Reasoning and matched evidence:**")
                for q in qs:
                    # show which disease the question targeted and the evidence sentence
                    st.write(f"- Question: {q['q']}  — targeted: {q['target']}  — evidence: {q['evidence']}")

                # show full details for both
                with st.expander(f"Details: {d_a}"):
                    st.json(next(x['details'] for x in disease_data if x['name']==d_a))
                with st.expander(f"Details: {d_b}"):
                    st.json(next(x['details'] for x in disease_data if x['name']==d_b))

    else:
        st.info("No distinguishing questions could be generated for this pair.")

else:
    st.info("Select two different diseases to begin disambiguation.")
