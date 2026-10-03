import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="PawVision AI | Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cats_dogs_model.h5"
MODEL_PATH_KERAS = BASE_DIR / "models" / "cats_dogs_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
HISTORY_IMG = BASE_DIR / "models" / "training_history.png"
CM_IMG = BASE_DIR / "models" / "confusion_matrix.png"
SAMPLES_DIR = BASE_DIR / "sample_images"
DATASET_DIR = BASE_DIR / "dataset"
IMG_SIZE = (224, 224)
EMOJI = {"cat": "🐱", "dog": "🐶"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px;}

.stApp {
    background:
        radial-gradient(900px 500px at 8% -10%, rgba(139,92,246,0.18), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,0.12), transparent 55%),
        #0B1020;
}

.hero {
    position: relative; overflow: hidden;
    padding: 2.6rem 2.4rem; border-radius: 24px;
    background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 45%, #0891B2 100%);
    box-shadow: 0 20px 60px rgba(79,70,229,0.35);
    margin-bottom: 1.4rem;
}
.hero::after {
    content: "🐾"; position: absolute; right: 2rem; top: 0.4rem;
    font-size: 9rem; opacity: 0.13; transform: rotate(14deg);
}
.hero .badge {
    display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
    background: rgba(255,255,255,0.16); color: #fff; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.04em; margin: 0 0.4rem 0.9rem 0;
}
.hero h1 {color: #fff; font-size: 2.7rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;}
.hero p {color: rgba(255,255,255,0.9); font-size: 1.05rem; margin: 0.6rem 0 0 0; max-width: 680px;}

.metric-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 1.1rem 1.25rem; height: 100%;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: transform .2s ease, border-color .2s ease;
}
.metric-card:hover {transform: translateY(-3px); border-color: rgba(139,92,246,0.6);}
.metric-label {color: #9AA3C7; font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;}
.metric-value {color: #fff; font-size: 1.8rem; font-weight: 800; margin-top: 0.2rem; letter-spacing: -0.01em;}
.metric-sub {color: #7C86AD; font-size: 0.8rem; margin-top: 0.1rem;}

.card {
    background: rgba(20,26,46,0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.3rem 1.5rem; height: 100%;
    box-shadow: 0 10px 40px rgba(0,0,0,0.28);
    color: #C9D0EC; line-height: 1.65; font-size: 0.95rem;
}
.card b {color: #fff;}
.section-title {color: #fff; font-weight: 700; font-size: 1.15rem; margin: 0 0 0.6rem 0;}
.section-sub {color: #9AA3C7; font-size: 0.92rem; margin: -0.3rem 0 1rem 0;}

.step {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 1.1rem 1.2rem; height: 100%;
}
.step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 2rem; height: 2rem; border-radius: 50%; margin-bottom: 0.6rem;
    background: linear-gradient(135deg, #8B5CF6, #22D3EE); color: #fff; font-weight: 800;
}
.step .t {color: #fff; font-weight: 700; margin-bottom: 0.2rem;}
.step .d {color: #A9B2D6; font-size: 0.9rem; line-height: 1.55;}

.verdict {border-radius: 18px; padding: 1.3rem 1.4rem; margin-bottom: 1.1rem; display: flex; align-items: center; gap: 1rem;}
.verdict.cat {background: linear-gradient(135deg, rgba(139,92,246,0.28), rgba(139,92,246,0.08)); border: 1px solid rgba(139,92,246,0.5);}
.verdict.dog {background: linear-gradient(135deg, rgba(34,211,238,0.26), rgba(34,211,238,0.07)); border: 1px solid rgba(34,211,238,0.5);}
.verdict.unsure {background: linear-gradient(135deg, rgba(251,191,36,0.22), rgba(251,191,36,0.06)); border: 1px solid rgba(251,191,36,0.5);}
.verdict .emoji {font-size: 3rem; line-height: 1;}
.verdict .title {color: #fff; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em;}
.verdict .sub {color: #B8C0E0; font-size: 0.9rem;}

.explain {
    border-left: 4px solid #8B5CF6; background: rgba(139,92,246,0.08);
    border-radius: 0 14px 14px 0; padding: 0.9rem 1.1rem; color: #D5DAF2;
    font-size: 0.93rem; line-height: 1.6; margin: 0.8rem 0;
}
.explain b {color: #fff;}

.stProgress > div > div > div > div {background: linear-gradient(90deg, #8B5CF6, #22D3EE);}
[data-testid="stFileUploader"] section {
    background: rgba(139,92,246,0.06); border: 2px dashed rgba(139,92,246,0.45);
    border-radius: 18px; padding: 1.6rem;
}
[data-testid="stFileUploader"] section:hover {background: rgba(139,92,246,0.12);}
[data-testid="stSidebar"] {background: #0E1428; border-right: 1px solid rgba(255,255,255,0.06);}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
.stTabs [data-baseweb="tab"] {border-radius: 12px; padding: 0.55rem 1.1rem; background: rgba(255,255,255,0.04);}
.stTabs [aria-selected="true"] {background: rgba(139,92,246,0.22) !important;}
.footer {text-align: center; color: #6B7597; font-size: 0.82rem; margin-top: 2.5rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model loading (with first-run auto-training fallback)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model_once():
    """First-run setup: trains the model from dataset/ if no saved model exists."""
    os.chdir(BASE_DIR)
    import train

    train.main([
        "--data_dir", str(BASE_DIR / "dataset"),
        "--batch_size", "8",
        "--epochs", "10",
        "--fine_tune_epochs", "0",
    ])
    return True


@st.cache_resource(show_spinner="Loading deep learning model...")
def load_model():
    import tensorflow as tf

    for path in (MODEL_PATH, MODEL_PATH_KERAS):
        if path.exists():
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue
    return None


@st.cache_data
def load_json(path_str):
    path = Path(path_str)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def dataset_files():
    """Return {'cat': [paths], 'dog': [paths]} found in dataset/."""
    found = {"cat": [], "dog": []}
    if not DATASET_DIR.exists():
        return found
    for p in sorted(DATASET_DIR.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        hint = f"{p.parent.name} {p.stem}".lower()
        if "cat" in hint:
            found["cat"].append(str(p))
        elif "dog" in hint:
            found["dog"].append(str(p))
    return found


def predict(model, image, class_names):
    img = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)
    p_second = float(model.predict(arr, verbose=0)[0][0])
    return {class_names[0]: 1.0 - p_second, class_names[1]: p_second}


def metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{html.escape(str(label))}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-sub">{html.escape(str(sub))}</div></div>""",
        unsafe_allow_html=True,
    )


def step_card(num, title, text):
    st.markdown(
        f"""<div class="step"><div class="num">{num}</div>
        <div class="t">{title}</div><div class="d">{text}</div></div>""",
        unsafe_allow_html=True,
    )


def sample_files():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMG_EXT)


model = load_model()
class_names = load_json(str(CLASSES_PATH)) or ["cat", "dog"]
metrics = load_json(str(METRICS_PATH)) or {}
ds = dataset_files()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐾 PawVision AI")
    st.caption("Deep learning image classifier")
    st.divider()

    st.markdown("### ⚙️ Controls")
    threshold = st.slider(
        "Confidence threshold", 50, 99, 70, 1, format="%d%%",
        help="If the model's top confidence is below this value, the result is shown as "
             "'not sure' instead of a firm answer. Higher = stricter.",
    )
    st.caption("Below this confidence, the app says *not sure* instead of guessing.")
    show_table = st.toggle("Show probability table", value=False,
                           help="Shows the exact probabilities as a small table.")

    samples = sample_files()
    sample_choice = "None"
    if samples:
        st.markdown("### 🖼️ Sample images")
        sample_choice = st.selectbox("Try a sample", ["None"] + [p.name for p in samples],
                                     help="No photo handy? Pick a demo image to test the model.")
    st.divider()

    st.markdown("### 📘 Quick guide")
    st.markdown(
        "1. Upload a cat or dog photo (or pick a sample)\n"
        "2. Read the verdict and confidence bars\n"
        "3. Open **Model Insights** to see how reliable the model is"
    )
    st.divider()

    st.markdown("### 🧠 Model")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Input:** 224 × 224 RGB")
    st.write("**Output:** Sigmoid (Cat / Dog)")
    st.write("**Framework:** TensorFlow / Keras")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero">
    <span class="badge">DEEP LEARNING</span><span class="badge">TRANSFER LEARNING</span><span class="badge">STREAMLIT</span>
    <h1>Cats vs Dogs Classifier</h1>
    <p>Upload a photo and a MobileNetV2 neural network tells you whether it shows a cat or a dog,
    with a clear confidence breakdown and an explanation of what the numbers mean.</p></div>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Top metric cards
# ----------------------------------------------------------------------------
acc = metrics.get("test_accuracy")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Test Accuracy", f"{acc:.1%}" if acc is not None else "—", "On unseen test images")
with c2:
    metric_card("Training Images", f"{metrics['train_images']:,}" if metrics else "—",
                f"{metrics.get('total_images', 0):,} images in total" if metrics else "Model not trained yet")
with c3:
    metric_card("Parameters", f"{metrics['parameters'] / 1e6:.2f}M" if metrics else "—", "MobileNetV2 + new head")
with c4:
    metric_card("Input Size", "224×224", "RGB, scaled to [-1, 1]")

st.write("")

# ----------------------------------------------------------------------------
# No model yet -> auto-train once (or show a clear error)
# ----------------------------------------------------------------------------
if model is None:
    if DATASET_DIR.exists() and not st.session_state.get("auto_trained"):
        st.info(
            "First-time setup: no trained model was found, so the app is training one from the "
            "`dataset/` folder. This takes a few minutes. Please keep this page open."
        )
        with st.spinner("Training the deep learning model... (about 2-5 minutes)"):
            try:
                train_model_once()
            except Exception as exc:
                st.error(f"Automatic training failed: {exc}")
                st.stop()
        st.session_state["auto_trained"] = True
        load_model.clear()
        load_json.clear()
        st.rerun()
    st.error(
        "Model file not found (or could not be loaded) in the `models/` folder. "
        "Run `python train.py` (or train on Google Colab) and commit the generated model "
        "files to your repository."
    )
    st.stop()

tab_classify, tab_data, tab_insights, tab_how = st.tabs(
    ["🔍 Classify", "🗂️ Dataset", "📊 Model Insights", "🧠 How it works"]
)

# ----------------------------------------------------------------------------
# Tab 1: Classify
# ----------------------------------------------------------------------------
with tab_classify:
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        step_card(1, "Add a photo", "Drag and drop a JPG or PNG below, or choose a demo image from the sidebar.")
    with s2:
        step_card(2, "Model analyses it", "The image is resized to 224×224 and passed through the neural network.")
    with s3:
        step_card(3, "Read the result", "You get a verdict, a confidence bar for each class and a plain-language explanation.")

    st.write("")
    uploaded = st.file_uploader(
        "Drag and drop an image here", type=["jpg", "jpeg", "png"],
        help="JPG or PNG, up to 10 MB. Best results: one clearly visible cat or dog.",
    )

    image = None
    source_name = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()
            source_name = uploaded.name
        except Exception:
            st.error("This file could not be read as an image. Please try another JPG or PNG.")
    elif sample_choice != "None":
        image = Image.open(SAMPLES_DIR / sample_choice)
        source_name = sample_choice

    st.write("")
    if image is None:
        st.markdown(
            """<div class="card" style="text-align:center; padding: 2.4rem;">
            <div style="font-size:3rem;">📸</div>
            <div class="section-title" style="margin-top:0.5rem;">No image yet</div>
            <div>Upload a cat or dog photo above, or pick a sample from the sidebar.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing image..."):
            probs = predict(model, image, class_names)
        top_class = max(probs, key=probs.get)
        other_class = [c for c in class_names if c != top_class][0]
        top_conf = probs[top_class]
        confident = top_conf * 100 >= threshold

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown('<div class="section-title">Your image</div>', unsafe_allow_html=True)
            st.image(image, caption=f"{source_name}  •  {image.size[0]}×{image.size[1]} px",
                     use_container_width=True)
            if min(image.size) < 100:
                st.warning("This image is very small, so the prediction may be less reliable.")

        with right:
            if confident:
                css = top_class if top_class in ("cat", "dog") else "cat"
                emoji = EMOJI.get(top_class, "🐾")
                title = f"It's a {top_class.title()}!"
                sub = f"Predicted with {top_conf:.1%} confidence"
            else:
                css, emoji = "unsure", "🤔"
                title = "Not sure about this one"
                sub = f"Best guess: {top_class.title()} ({top_conf:.1%}), below your {threshold}% threshold"
            st.markdown(
                f"""<div class="verdict {css}"><div class="emoji">{emoji}</div>
                <div><div class="title">{html.escape(title)}</div><div class="sub">{html.escape(sub)}</div></div></div>""",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Confidence breakdown</div>', unsafe_allow_html=True)
            for name in class_names:
                st.progress(float(probs[name]), text=f"{EMOJI.get(name, '')} {name.title()}: {probs[name]:.1%}")

            st.write("")
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Prediction", top_class.title(), "Most likely class")
            with m2:
                metric_card("Confidence", f"{top_conf:.1%}", "Top probability")
            with m3:
                margin = abs(probs[class_names[0]] - probs[class_names[1]])
                metric_card("Margin", f"{margin:.1%}", "Gap between classes")

        # Plain-language explanation of the result
        if top_conf >= 0.90:
            level = "very confident"
        elif top_conf >= 0.75:
            level = "fairly confident"
        else:
            level = "unsure"
        caution = "" if confident else "Because the top confidence is below your threshold, treat this answer with caution."
        st.markdown(
            f"""<div class="explain"><b>How to read this result:</b> the model is <b>{level}</b>.
            It gives <b>{top_conf:.1%}</b> to <b>{html.escape(top_class)}</b> and
            <b>{probs[other_class]:.1%}</b> to <b>{html.escape(other_class)}</b>; the two always add up to 100%.
            The <b>margin</b> is the gap between them: the bigger it is, the clearer the decision.
            {caution}</div>""",
            unsafe_allow_html=True,
        )

        with st.expander("Tips for better predictions"):
            st.markdown(
                "- Use a photo where the animal is clearly visible and fills most of the frame.\n"
                "- Avoid heavy blur, very dark lighting, or several animals in one picture.\n"
                "- The model only knows **cats and dogs**: any other object will still be forced into one of the two classes.\n"
                "- The model was trained on a small dataset, so occasional mistakes are normal."
            )

        if show_table:
            df = pd.DataFrame(
                {"Class": [c.title() for c in class_names],
                 "Probability": [round(probs[c], 6) for c in class_names]}
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: Dataset
# ----------------------------------------------------------------------------
with tab_data:
    n_cat, n_dog = len(ds["cat"]), len(ds["dog"])
    st.markdown('<div class="section-title">The data behind the model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The model learns from labelled photos. Images come from the '
        '<a href="https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset" target="_blank">'
        'Cats and Dogs Mini Dataset on Kaggle</a>.</div>',
        unsafe_allow_html=True,
    )
    if n_cat + n_dog == 0:
        st.info("The `dataset/` folder was not found in this deployment.")
    else:
        d1, d2, d3 = st.columns(3)
        with d1:
            metric_card("Total images", f"{n_cat + n_dog}", "In the dataset folder")
        with d2:
            metric_card("Cat images", f"{n_cat}", "Class label 0")
        with d3:
            metric_card("Dog images", f"{n_dog}", "Class label 1")

        st.write("")
        g1, g2 = st.columns([1, 1.4], gap="large")
        with g1:
            st.markdown('<div class="section-title">Class balance</div>', unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame({"Images": [n_cat, n_dog]}, index=["Cats", "Dogs"]))
            st.caption("A balanced dataset stops the model from favouring one class.")
        with g2:
            st.markdown(
                """<div class="card"><div class="section-title">How the data is used</div>
                <b>Train (about 70%)</b>: the images the model learns from.<br>
                <b>Validation (about 15%)</b>: used during training to decide when to stop.<br>
                <b>Test (about 15%)</b>: never seen in training; gives the honest final accuracy.<br><br>
                The split is <b>stratified</b>, so cats and dogs appear in every part.
                Because this is a <i>mini</i> dataset, the test set is only a few images, so the accuracy
                can vary between training runs.</div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown('<div class="section-title">Image gallery</div>', unsafe_allow_html=True)
        gc1, gc2 = st.columns(2, gap="large")
        for col, key, label in ((gc1, "cat", "🐱 Cats"), (gc2, "dog", "🐶 Dogs")):
            with col:
                st.markdown(f"**{label}**")
                shown = ds[key][:6]
                if shown:
                    rows = [shown[i:i + 3] for i in range(0, len(shown), 3)]
                    for row in rows:
                        cols = st.columns(3)
                        for c, p in zip(cols, row):
                            c.image(p, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: Model Insights
# ----------------------------------------------------------------------------
with tab_insights:
    if not metrics:
        st.info("Training metrics are not available yet. They are created when the model is trained.")
    else:
        n_test = metrics["test_images"]
        n_right = int(round(metrics["test_accuracy"] * n_test))
        st.markdown('<div class="section-title">How good is the model?</div>', unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Test Accuracy", f"{metrics['test_accuracy']:.1%}", "Share of test images correct")
        with b:
            metric_card("Validation Accuracy", f"{metrics['val_accuracy']:.1%}", "Best during training")
        with c:
            metric_card("Test Loss", f"{metrics['test_loss']:.3f}", "Lower is better")
        with d:
            metric_card("Test Images", f"{n_test:,}", "Never used for training")

        st.markdown(
            f"""<div class="explain"><b>In plain words:</b> the model was shown <b>{n_test}</b> photos it had never
            seen and got about <b>{n_right}</b> of them right ({metrics['test_accuracy']:.1%}).
            With so few test images, one extra mistake changes the percentage a lot, so treat it as an
            estimate rather than an exact score.</div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Per-class performance</div>', unsafe_allow_html=True)
        report_df = pd.DataFrame(
            {
                "Class": [c.title() for c in class_names],
                "Precision": [metrics["precision"][c] for c in class_names],
                "Recall": [metrics["recall"][c] for c in class_names],
                "F1-score": [metrics["f1"][c] for c in class_names],
            }
        )
        st.dataframe(
            report_df.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}"}),
            hide_index=True, use_container_width=True,
        )
        with st.expander("What do precision, recall and F1 mean?"):
            st.markdown(
                "- **Precision**: when the model says *dog*, how often is it really a dog?\n"
                "- **Recall**: of all the real dogs, how many did the model find?\n"
                "- **F1-score**: one number that balances precision and recall (1.0 = perfect).\n\n"
                "The same three numbers are shown for cats."
            )

        g1, g2 = st.columns([1.5, 1], gap="large")
        with g1:
            if HISTORY_IMG.exists():
                st.markdown('<div class="section-title">Training history</div>', unsafe_allow_html=True)
                st.image(str(HISTORY_IMG), use_container_width=True)
                st.caption("Accuracy should rise and loss should fall. If the validation line "
                           "stays far from the training line, the model may be overfitting.")
        with g2:
            if CM_IMG.exists():
                st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
                st.image(str(CM_IMG), use_container_width=True)
                st.caption("Rows are the real class, columns are the predicted class. "
                           "Numbers on the diagonal are correct predictions.")

# ----------------------------------------------------------------------------
# Tab 4: How it works
# ----------------------------------------------------------------------------
with tab_how:
    st.markdown('<div class="section-title">From photo to prediction</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        step_card(1, "Preprocess", "The photo is fixed for rotation, converted to RGB and resized to 224×224 pixels.")
    with h2:
        step_card(2, "Rescale", "Pixel values (0 to 255) are mapped to the range -1 to 1, which MobileNetV2 expects.")
    with h3:
        step_card(3, "Extract features", "MobileNetV2, pre-trained on ImageNet, turns the image into 1280 numbers describing shapes, textures and patterns.")
    with h4:
        step_card(4, "Classify", "A final neuron outputs the probability of 'dog'. Cat probability is 1 minus that.")

    st.write("")
    k1, k2 = st.columns(2, gap="large")
    with k1:
        st.markdown(
            """<div class="card"><div class="section-title">What is transfer learning?</div>
            Training a deep network from scratch needs millions of images. Instead, we reuse
            <b>MobileNetV2</b>, which already learned to see edges, fur, eyes and shapes from ImageNet.
            We only train a small new layer on top for our two classes, which works well even with a
            small dataset and trains quickly on a normal CPU.</div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """<div class="card"><div class="section-title">How was it trained?</div>
            <b>Data augmentation</b> (random flips, rotations, zoom and contrast) gives the model many
            slightly different versions of each photo, which reduces overfitting.<br>
            <b>Phase 1</b>: only the new top layer is trained.<br>
            <b>Phase 2</b> (optional): the top layers of MobileNetV2 are fine-tuned with a very small learning rate.<br>
            <b>Early stopping</b> halts training when validation loss stops improving.</div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="section-title">Frequently asked questions</div>', unsafe_allow_html=True)
    with st.expander("Why does the app sometimes say 'not sure'?"):
        st.write("When the top confidence is below the threshold set in the sidebar, the app avoids "
                 "giving a firm answer. You can lower or raise the threshold to be less or more strict.")
    with st.expander("What happens if I upload something that is not a cat or dog?"):
        st.write("The model only knows two classes, so it will still pick the closer one. "
                 "Low confidence is often a hint that the picture is neither.")
    with st.expander("Why can the accuracy change when the model is retrained?"):
        st.write("The dataset is small and the test set has only a few images, so a single image can move "
                 "the percentage a lot. More images give steadier results.")
    with st.expander("What is the sigmoid output?"):
        st.write("A number between 0 and 1. Close to 1 means 'dog', close to 0 means 'cat', "
                 "and values near 0.5 mean the model is undecided.")

st.markdown('<div class="footer">Built with TensorFlow, Keras and Streamlit • PawVision AI</div>', unsafe_allow_html=True)
