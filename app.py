"""Application Streamlit unique : accès à tous les modèles Deep Learning du devoir.

Lancer en local :  streamlit run app.py
Les modèles entraînés (.keras) et leurs fichiers de prétraitement (.json) se trouvent dans le dossier
`models/` à côté de ce fichier.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
IMG_SIZE = (64, 64)  # taille utilisée à l'entraînement des CNN radiographies

st.set_page_config(page_title="Modèles Deep Learning – Master UADB", page_icon="🧠", layout="centered")


# ----------------------------------------------------------------------------------------------
# Outils communs
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_keras_model(filename: str):
    """Charge un modèle Keras une seule fois (None si le fichier est absent)."""
    import tensorflow as tf

    path = MODELS_DIR / filename
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


def load_json(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def files_ok(*names: str) -> bool:
    """Affiche un message clair si un fichier du dossier models/ manque."""
    missing = [n for n in names if not (MODELS_DIR / n).exists()]
    if missing:
        st.error("Fichier(s) introuvable(s) dans `models/` : " + ", ".join(f"`{m}`" for m in missing))
        return False
    return True


# ----------------------------------------------------------------------------------------------
# TP1 : Bank Marketing (classification)
# ----------------------------------------------------------------------------------------------
def page_bank():
    st.header("TP1 : Souscription à un dépôt à terme")
    st.write("Classification (réseau de neurones dense) sur le jeu de données Bank Marketing. "
             "Hyperparamètres optimisés avec Keras Tuner.")
    if not files_ok("tp1_bank_model.keras", "tp1_bank_preproc.json"):
        return
    model = load_keras_model("tp1_bank_model.keras")
    prep = load_json("tp1_bank_preproc.json")
    feats, cats = prep["features"], prep["categorical"]
    defaults = {"age": 40, "duration": 200, "campaign": 1, "pdays": 999, "previous": 0}

    with st.form("form_bank"):
        values = {}
        cols = st.columns(2)
        for i, f in enumerate(feats):
            with cols[i % 2]:
                if f in cats:
                    values[f] = st.selectbox(f, cats[f])
                else:
                    values[f] = st.number_input(f, value=float(defaults.get(f, 0)), step=1.0)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        # même encodage et même normalisation (RobustScaler) qu'à l'entraînement
        row = [cats[f].index(values[f]) if f in cats else float(values[f]) for f in feats]
        x = (np.array(row, dtype=np.float32) - np.array(prep["center"], dtype=np.float32)) / np.array(
            prep["scale"], dtype=np.float32)
        p_yes = float(model.predict(x[None, :], verbose=0)[0][0])
        if p_yes >= 0.5:
            st.success(f"Le client va **souscrire** (probabilité {p_yes:.1%})")
        else:
            st.warning(f"Le client ne va **pas souscrire** (probabilité de souscription {p_yes:.1%})")


# ----------------------------------------------------------------------------------------------
# TP2 : Student Performance (régression)
# ----------------------------------------------------------------------------------------------
def page_student():
    st.header("TP2 : Performance des étudiants")
    st.write("Régression (réseau de neurones dense) de l'indice de performance d'un étudiant. "
             "Hyperparamètres optimisés avec Keras Tuner.")
    if not files_ok("tp2_student_model.keras", "tp2_student_preproc.json"):
        return
    model = load_keras_model("tp2_student_model.keras")
    prep = load_json("tp2_student_preproc.json")
    feats, cats, ranges = prep["features"], prep["categorical"], prep.get("ranges", {})

    with st.form("form_student"):
        values = {}
        for f in feats:
            if f in cats:
                values[f] = st.selectbox(f, cats[f])
            else:
                lo, hi = ranges.get(f, [0.0, 100.0])
                values[f] = st.number_input(f, min_value=float(lo), max_value=float(hi), value=float(round((lo + hi) / 2)),
                                            step=1.0)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        row = [cats[f].index(values[f]) if f in cats else float(values[f]) for f in feats]
        x = (np.array(row, dtype=np.float32) - np.array(prep["center"], dtype=np.float32)) / np.array(
            prep["scale"], dtype=np.float32)
        pred = float(model.predict(x[None, :], verbose=0)[0][0])
        pred = min(max(pred, 0.0), 100.0)
        st.metric("Indice de performance prédit", f"{pred:.1f} / 100")


# ----------------------------------------------------------------------------------------------
# TP3 : CIFAR-10 (CNN, VGG16, DenseNet121)
# ----------------------------------------------------------------------------------------------
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def prep_cnn(x):  # pixels 0-255 -> 0-1
    return x.astype("float32") / 255.0


def prep_vgg16(x):  # RGB -> BGR puis moyenne ImageNet retirée
    return x.astype("float32")[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype="float32")


def prep_densenet121(x):  # 0-1 puis moyenne / écart-type ImageNet
    x = x.astype("float32") / 255.0
    return (x - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array([0.229, 0.224, 0.225], dtype="float32")


TP3_MODELS = {
    "CNN (baseline)": ("tp3_cnn_model.keras", prep_cnn),
    "VGG16": ("tp3_vgg16_model.keras", prep_vgg16),
    "DenseNet121": ("tp3_densenet121_model.keras", prep_densenet121),
}


def page_cifar():
    st.header("TP3 : Classification d'images CIFAR-10")
    st.write("Choisissez un modèle (CNN du TP, VGG16 ou DenseNet121, tous avec hyperparamètres optimisés par Keras Tuner) "
             "puis téléversez une image. Classes : " + ", ".join(CIFAR_CLASSES) + ".")
    name = st.selectbox("Modèle", list(TP3_MODELS.keys()))
    filename, prep_fn = TP3_MODELS[name]
    if not files_ok(filename):
        return
    model = load_keras_model(filename)

    uploaded = st.file_uploader("Téléversez une image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="cifar_upload")
    if uploaded is None:
        return
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image téléversée", width="stretch")

    x = np.asarray(image.resize((32, 32)), dtype="float32")  # pixels 0-255, comme à l'entraînement
    probs = model.predict(prep_fn(x)[None, ...], verbose=0)[0]
    top = np.argsort(probs)[::-1][:4]
    st.success(f"Classe prédite ({name}) : **{CIFAR_CLASSES[top[0]]}** (confiance {probs[top[0]]:.1%})")
    st.bar_chart(pd.DataFrame({"Probabilité": [float(probs[i]) for i in top]}, index=[CIFAR_CLASSES[i] for i in top]))


# ----------------------------------------------------------------------------------------------
# Radiographies : tuberculose et pneumonie
# ----------------------------------------------------------------------------------------------
def preprocess_xray(image: Image.Image) -> np.ndarray:
    """Même prétraitement qu'à l'entraînement : RGB, 64x64, interpolation NEAREST, pixels /255."""
    image = image.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)


def page_xray(title: str, model_file: str, positive_label: str, description: str):
    """Page générique pour un CNN binaire sur radiographie (classe 1 = maladie)."""
    st.header(title)
    st.write(description)
    if not files_ok(model_file):
        return
    model = load_keras_model(model_file)

    uploaded = st.file_uploader("Téléversez une radiographie thoracique (PNG, JPG)", type=["png", "jpg", "jpeg"],
                                key=model_file)
    if uploaded is None:
        return

    image = Image.open(uploaded)
    st.image(image, caption="Image téléversée", width="stretch")

    p_pos = float(model.predict(preprocess_xray(image), verbose=0)[0][0])  # P(classe 1)
    label = positive_label if p_pos >= 0.5 else "Normal"
    confidence = p_pos if p_pos >= 0.5 else 1.0 - p_pos

    if p_pos >= 0.5:
        st.error(f"Résultat : **{label}** (confiance {confidence:.1%})")
    else:
        st.success(f"Résultat : **{label}** (confiance {confidence:.1%})")
    st.progress(p_pos, text=f"Probabilité {positive_label} : {p_pos:.1%}")
    st.caption("Outil pédagogique : il ne remplace pas un diagnostic médical.")


# ----------------------------------------------------------------------------------------------
# Accueil et navigation
# ----------------------------------------------------------------------------------------------
def page_accueil():
    st.title("🧠 Modèles Deep Learning")
    st.write("Application unique du devoir de Deep Learning (Master 1, UADB). "
             "Choisissez un modèle dans le menu de gauche.")
    st.markdown(
        "- **TP1** : classification, souscription bancaire (réseau dense)\n"
        "- **TP2** : régression, performance des étudiants (réseau dense)\n"
        "- **TP3** : images CIFAR-10 (CNN, VGG16, DenseNet121)\n"
        "- **Tuberculose** : CNN sur radiographies\n"
        "- **Pneumonie** : CNN sur radiographies\n\n"
        "Tous les modèles ont leurs hyperparamètres optimisés avec Keras Tuner."
    )
    present = sorted(p.name for p in MODELS_DIR.glob("*.keras")) if MODELS_DIR.exists() else []
    st.sidebar.caption("Modèles trouvés : " + (", ".join(present) if present else "aucun"))


PAGES = {
    "Accueil": page_accueil,
    "TP1 – Souscription bancaire": page_bank,
    "TP2 – Performance étudiante": page_student,
    "TP3 – Images CIFAR-10": page_cifar,
    "Tuberculose (radiographie)": lambda: page_xray(
        "Détection de la tuberculose", "tb_cnn_model.keras", "Tuberculose",
        "Téléversez une radiographie thoracique pour estimer la probabilité de tuberculose."),
    "Pneumonie (radiographie)": lambda: page_xray(
        "Détection de la pneumonie", "pneumonia_cnn_model.keras", "Pneumonie",
        "Téléversez une radiographie thoracique pour estimer la probabilité de pneumonie."),
}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[choice]()
