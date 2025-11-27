import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# =========================
#   CONFIGURACIÓN STREAMLIT
# =========================

st.set_page_config(
    page_title="Digimon TCG Price Predictor",
    page_icon="🃏",
    layout="wide"
)

# Estilo con más contraste
st.markdown(
    """
    <style>
    .stApp {
        background-color: #020617;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #f9fafb;
    }
    h1, h2, h3, h4, h5 {
        color: #f9fafb;
    }
    label, .stMarkdown, .stText, .stCaption, .stRadio, .stSelectbox {
        color: #e5e7eb !important;
    }
    /* Inputs claros con texto oscuro */
    input, textarea, select {
        color: #0f172a !important;
        background-color: #f9fafb !important;
    }
    .stTextInput>div>div>input,
    .stNumberInput input,
    .stSelectbox > div > div > select,
    .stTextArea textarea {
        color: #0f172a !important;
        background-color: #f9fafb !important;
    }
    /* Contenedor de las métricas de resultado */
    .metric-container {
        background-color: #0f172a;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
#   CARGA MODELOS & METADATA
# =========================

@st.cache_resource
def load_models_and_metadata():
    model_general = joblib.load("model_general_final.pkl")
    model_premium = joblib.load("model_premium_final.pkl")

    with open("metadata_final.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model_general, model_premium, metadata


@st.cache_data
def load_dataset(path: str = "digimon_cards_with_prices_full.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


model_general, model_premium, metadata = load_models_and_metadata()
df = load_dataset()

cat_cols = metadata["cat_cols"]
feature_columns = metadata["feature_columns"]
rarity_order_map = metadata["rarity_order_map"]
premium_threshold = metadata["premium_threshold"]
premium_rarities = metadata["premium_rarities"]

# =========================
#   OPCIONES DE SELECTBOX
# =========================

def get_options(col_name):
    """Devuelve lista de opciones únicas para una columna del dataset."""
    if df is not None and col_name in df.columns:
        return [""] + sorted(df[col_name].dropna().astype(str).unique())
    else:
        return [""]

code_options = get_options("code")
set_name_options = get_options("set_name")
color2_options = get_options("color2")
attribute_options = get_options("attribute")
form_options = get_options("form")
stage_options = get_options("stage")

# =========================
#   PREPROCESADO E INFERENCIA
# =========================

def preprocess_for_inference(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la lógica básica de features usada en entrenamiento
    y ordena las columnas como en feature_columns.
    """
    df_proc = df_input.copy()

    # Nunca usamos avg como feature
    if "avg" in df_proc.columns:
        df_proc = df_proc.drop(columns=["avg"])

    # Rarity + rarity_rank
    if "rarity" in df_proc.columns:
        df_proc["rarity"] = df_proc["rarity"].astype(str).str.upper()
        if "rarity_rank" not in df_proc.columns:
            df_proc["rarity_rank"] = df_proc["rarity"].map(rarity_order_map).fillna(-1).astype(int)

    # Fecha -> año/mes
    if "date_added" in df_proc.columns:
        df_proc["date_added"] = pd.to_datetime(df_proc["date_added"], errors="coerce")
        df_proc["year_added"] = df_proc["date_added"].dt.year
        df_proc["month_added"] = df_proc["date_added"].dt.month
        df_proc = df_proc.drop(columns=["date_added"])

    # Asegurar que todas las columnas de feature_columns existen
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = np.nan

    # Orden correcto
    df_proc = df_proc[feature_columns].copy()

    # Categóricas como en entrenamiento
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype(str).fillna("UNK")

    # Numéricas
    for col in df_proc.columns:
        if col not in cat_cols:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

    return df_proc


def hybrid_predict(df_input: pd.DataFrame) -> np.ndarray:
    """
    Modelo híbrido:
      - General en log-precio.
      - Si pred_general >= umbral -> pasa por modelo premium.
    """
    X = preprocess_for_inference(df_input)

    pred_log_general = model_general.predict(X)
    pred_general = np.expm1(pred_log_general)

    gate = pred_general >= premium_threshold
    final_pred = pred_general.copy()

    if gate.any():
        X_prem = X[gate]
        pred_log_premium = model_premium.predict(X_prem)
        pred_premium = np.expm1(pred_log_premium)
        final_pred[gate] = pred_premium

    return final_pred

# =========================
#   UI PRINCIPAL (SOLO PREDICCIÓN)
# =========================

st.title("🃏 Digimon TCG Price Predictor")
st.caption("Introduce las características de una carta y estima su precio `avg`, esté o no en el dataset.")

st.subheader("Introduce los datos de la carta")

with st.form("card_form"):
    col1, col2 = st.columns(2)

    with col1:
        name_x = st.text_input("Nombre de la carta (name_x)", value="")

        code = st.selectbox(
            "Código (BTx-xxx, EXx-xxx, etc.)",
            options=code_options,
            index=0
        )

        set_name = st.selectbox(
            "Set / expansión (set_name)",
            options=set_name_options,
            index=0
        )

        type_ = st.selectbox(
            "Tipo (type)",
            ["", "Digimon", "Tamer", "Option"],
            index=1  # por defecto Digimon
        )

        color = st.selectbox(
            "Color principal (color)",
            ["", "Red", "Blue", "Yellow", "Green", "Black", "Purple", "White"],
            index=0
        )

        color2 = st.selectbox(
            "Color secundario (color2)",
            options=color2_options,
            index=0
        )

        rarity = st.selectbox(
            "Rareza (rarity)",
            ["C", "U", "R", "SR", "P", "SEC"],
            index=3  # por defecto SR
        )

    with col2:
        level = st.number_input("Level", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
        play_cost = st.number_input("Play cost", min_value=0.0, max_value=20.0, step=1.0, value=0.0)
        evolution_cost = st.number_input("Evolution cost", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
        evolution_level = st.number_input("Evolution level", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
        dp = st.number_input("DP", min_value=0.0, max_value=20000.0, step=1000.0, value=0.0)

        attribute = st.selectbox(
            "Attribute (attribute)",
            options=attribute_options,
            index=0
        )

        form = st.selectbox(
            "Form (form)",
            options=form_options,
            index=0
        )

        stage = st.selectbox(
            "Stage (stage)",
            options=stage_options,
            index=0
        )

    st.markdown("**Texto de efectos (opcional)**")
    main_effect = st.text_area("Main effect (main_effect)", value="", height=80)
    source_effect = st.text_area("Source effect (source_effect)", value="", height=80)

    submitted = st.form_submit_button("Predecir precio")

if submitted:
    # Construimos un DataFrame de una fila con lo introducido
    card_data = {
        "name_x": [name_x],
        "type": [type_ if type_ != "" else np.nan],
        "level": [level if level != 0.0 else np.nan],
        "play_cost": [play_cost if play_cost != 0.0 else np.nan],
        "evolution_cost": [evolution_cost if evolution_cost != 0.0 else np.nan],
        "evolution_color": [np.nan],
        "evolution_level": [evolution_level if evolution_level != 0.0 else np.nan],
        "xros_req": [np.nan],
        "color": [color if color != "" else np.nan],
        "color2": [color2 if color2 != "" else np.nan],
        "digi_type": [np.nan],
        "digi_type2": [np.nan],
        "form": [form if form != "" else np.nan],
        "dp": [dp if dp != 0.0 else np.nan],
        "attribute": [attribute if attribute != "" else np.nan],
        "rarity": [rarity],
        "stage": [stage if stage != "" else np.nan],
        "main_effect": [main_effect if main_effect != "" else np.nan],
        "source_effect": [source_effect if source_effect != "" else np.nan],
        "alt_effect": [np.nan],
        "date_added": [pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")],
        "set_name": [set_name if set_name != "" else np.nan],
        "name_y": [f"{name_x} ({code})" if name_x and code else np.nan],
        "code": [code if code != "" else np.nan],
    }

    df_card = pd.DataFrame(card_data)

    # Predicción
    y_pred_card = hybrid_predict(df_card)[0]

    st.markdown("### Resultado de la predicción")
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Precio estimado (avg)", f"{y_pred_card:.2f} €")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("**Resumen de la carta introducida:**")
    st.json({k: v[0] for k, v in card_data.items()})

