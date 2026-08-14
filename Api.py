#------------------------------------------------------------------------------------------------
# APLICACIÓN STREAMLIT
# Predicción de Empleo Formal vs Informal
# Random Forest Classifier
#------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from joblib import load


#------------------------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
#------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Predicción de Empleo",
    page_icon="💼",
    layout="wide"
)


#------------------------------------------------------------------------------------------------
# DISEÑO CSS
#------------------------------------------------------------------------------------------------

st.markdown(
"""
<style>
/* Fondo general */
.stApp {
    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);
    color: white !important;
}

/* Formularios */
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
}

/* Botones */
div.stButton > button,
button[kind="primary"] {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    font-size: 18px;
    font-weight: 700;
    color: white !important;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    box-shadow: 0px 6px 18px rgba(37,99,235,0.35);
}

div.stButton > button:hover,
button[kind="primary"]:hover {
    background: linear-gradient(90deg,#7C3AED,#2563EB);
}

/* BARRA LATERAL */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #1E293B) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stAlert {
    background: rgba(37, 99, 235, 0.2) !important;
    border: 1px solid rgba(37, 99, 235, 0.3) !important;
}

section[data-testid="stSidebar"] h1 {
    color: #60A5FA !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] .stMarkdown strong {
    color: #93C5FD !important;
}

/* Tarjetas de resultado */
.result-card {
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 10px;
    box-shadow: 0px 10px 28px rgba(0,0,0,0.35);
}

.formal {
    background: linear-gradient(135deg,#16A34A,#22C55E);
    color: white !important;
}

.informal {
    background: linear-gradient(135deg,#DC2626,#EF4444);
    color: white !important;
}

/* Encabezado */
.header-card {
    background: linear-gradient(90deg,#0F172A,#1E3A8A,#2563EB);
    border-radius: 26px;
    padding: 32px;
    text-align: center;
    color: white !important;
    box-shadow: 0px 12px 35px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.12);
}

.header-card h1 {
    color: white !important;
    margin-bottom: 6px;
    font-size: 46px;
}

.header-card h3 {
    color: #E0F2FE !important;
    margin-top: 0;
    font-weight: 400;
}

.header-card p {
    margin: 6px 0;
    font-size: 17px;
    color: white !important;
}

hr {
    border: 1px solid rgba(255,255,255,0.15);
}

.stSubheader {
    color: white !important;
}

/* Alertas */
div[data-testid="stAlert"] {
    color: white !important;
}

div[data-testid="stAlert"] div {
    color: white !important;
}

.stSuccess {
    background: rgba(22, 163, 74, 0.2) !important;
    border: 1px solid rgba(22, 163, 74, 0.3) !important;
}

.stWarning {
    background: rgba(234, 179, 8, 0.2) !important;
    border: 1px solid rgba(234, 179, 8, 0.3) !important;
}

.stInfo {
    background: rgba(37, 99, 235, 0.2) !important;
    border: 1px solid rgba(37, 99, 235, 0.3) !important;
}

.stCaption {
    color: #94A3B8 !important;
}

/* ============================================
   CORRECCIÓN: ESTILOS PARA SELECTBOX E INPUTS
   ============================================ */

/* Selectbox - texto y fondo oscuro */
div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #E2E8F0 !important;
}

div[data-baseweb="select"] div {
    color: #E2E8F0 !important;
}

div[data-baseweb="select"] input {
    color: #E2E8F0 !important;
    background-color: rgba(15, 23, 42, 0.5) !important;
}

/* Dropdown menu */
div[data-baseweb="popover"] {
    background-color: #1E293B !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

div[data-baseweb="popover"] * {
    color: #E2E8F0 !important;
}

div[data-baseweb="popover"] li {
    background-color: #1E293B !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #334155 !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    color: #E2E8F0 !important;
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] label {
    color: #94A3B8 !important;
}

/* Etiquetas de los campos */
.stSelectbox label, .stNumberInput label {
    color: #94A3B8 !important;
}

/* Estilos para los títulos de las barras de progreso */
.stMarkdown h4 {
    color: white !important;
}

</style>
""",
unsafe_allow_html=True
)


#------------------------------------------------------------------------------------------------
# CARGAR MODELO
#------------------------------------------------------------------------------------------------

clf = load("modelo_empleo.joblib")


#------------------------------------------------------------------------------------------------
# OPCIONES
#------------------------------------------------------------------------------------------------

sexo_options = ["Hombre", "Mujer"]

etnia_options = ["Mestizo", "Quechua", "Aymara", "Otro"]

lengua_options = ["Castellano", "Quechua", "Otra"]

area_options = ["Urbana", "Rural"]

educacion_options = [
    "Sin educación",
    "Primaria",
    "Secundaria",
    "Superior",
    "Universidad",
    "Posgrado"
]

ocupacion_options = [
    "Empleado",
    "Independiente",
    "Trabajador familiar",
    "Otro"
]

tam_empresa_options = [
    "Micro",
    "Pequeña",
    "Mediana",
    "Grande"
]


#------------------------------------------------------------------------------------------------
# BARRA LATERAL
#------------------------------------------------------------------------------------------------

st.sidebar.title("📌 Información del Proyecto")

st.sidebar.markdown(
    """
**Algoritmo:** Random Forest Classifier

**Objetivo:** Predecir si una persona pertenece al empleo formal o informal.

**Variables utilizadas:**
- Sexo
- Edad
- Etnia
- Lengua
- Área
- Educación
- Ocupación
- Tamaño de empresa
- Horas trabajadas
"""
)

st.sidebar.info(
    "Aplicación desarrollada en Streamlit para despliegue web de modelos de Machine Learning."
)


#------------------------------------------------------------------------------------------------
# ENCABEZADO PRINCIPAL - AGREGADO EINER SOTOMAYOR HUAMAN
#------------------------------------------------------------------------------------------------

st.markdown(
"""
<div class="header-card">

<h1>💼 Predicción del Empleo Formal e Informal</h1>
<h3>Random Forest Classifier</h3>

<hr>

<p><b>Unidad Didáctica:</b><br>
612491 - MACHINE LEARNING EN PRODUCCIÓN - DESPLIEGUE WEB</p>

<p><b>Docente:</b><br>
Orlando Advíncula Zeballos</p>

<p><b>Elaborado por:</b><br>
Bach. Yubber Franklin Soria Ccarhuas<br>
Einer Sotomayor Huaman</p>

</div>
""",
unsafe_allow_html=True
)

st.write("")


#------------------------------------------------------------------------------------------------
# FORMULARIO
#------------------------------------------------------------------------------------------------

with st.form("form_empleo"):

    st.subheader("📝 Ingreso de datos de la persona")

    col1, col2 = st.columns(2)

    with col1:

        sexo = st.selectbox("👤 Sexo", sexo_options)

        edad = st.number_input(
            "🎂 Edad",
            min_value=0,
            max_value=100,
            value=30
        )

        etnia = st.selectbox("🌎 Etnia", etnia_options)

        lengua = st.selectbox("🗣 Lengua", lengua_options)

        area = st.selectbox("🏙 Área", area_options)

    with col2:

        educacion = st.selectbox("🎓 Educación", educacion_options)

        ocupacion = st.selectbox("💼 Ocupación", ocupacion_options)

        tam_empresa = st.selectbox(
            "🏢 Tamaño de empresa",
            tam_empresa_options
        )

        horas_tr = st.number_input(
            "⏰ Horas trabajadas por semana",
            min_value=0,
            max_value=120,
            value=48
        )

    boton = st.form_submit_button("🚀 Realizar Predicción")


#------------------------------------------------------------------------------------------------
# PREDICCIÓN
#------------------------------------------------------------------------------------------------

if boton:

    datos_persona = pd.DataFrame({

        "Sexo": [sexo],
        "Edad": [edad],
        "Etnia": [etnia],
        "Lengua": [lengua],
        "Area": [area],
        "Educacion": [educacion],
        "Ocupacion": [ocupacion],
        "Tam_empresa": [tam_empresa],
        "Horas_Tr": [horas_tr]

    })

    prediccion = clf.predict(datos_persona)[0]

    probabilidades = clf.predict_proba(datos_persona)[0]

    clases = list(clf.classes_)

    prob_formal = probabilidades[clases.index("Formal")]
    prob_informal = probabilidades[clases.index("Informal")]

    st.write("")

    # RESULTADO PRINCIPAL

    if prediccion == "Formal":

        st.markdown(
            f"""
            <div class="result-card formal">
            ✅ EMPLEO FORMAL
            <br>
            <span style="font-size:18px;font-weight:400;">
            Alta probabilidad de inserción laboral formal
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-card informal">
            ⚠️ EMPLEO INFORMAL
            <br>
            <span style="font-size:18px;font-weight:400;">
            Alta probabilidad de inserción laboral informal
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # MÉTRICAS - USANDO HTML PERSONALIZADO EN LUGAR DE st.metric()

    st.subheader("📊 Probabilidades estimadas")

    col1, col2 = st.columns(2)

    with col1:
        # Usamos HTML personalizado en lugar de st.metric()
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); padding: 18px; border-radius: 18px; box-shadow: 0px 8px 22px rgba(0,0,0,0.25); text-align: center;">
                <p style="color: #93C5FD; font-weight: 500; margin: 0; font-size: 16px;">Empleo Formal</p>
                <p style="color: #FFFFFF; font-size: 38px; font-weight: 700; margin: 5px 0;">{prob_formal*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); padding: 18px; border-radius: 18px; box-shadow: 0px 8px 22px rgba(0,0,0,0.25); text-align: center;">
                <p style="color: #93C5FD; font-weight: 500; margin: 0; font-size: 16px;">Empleo Informal</p>
                <p style="color: #FFFFFF; font-size: 38px; font-weight: 700; margin: 5px 0;">{prob_informal*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # BARRAS DE PROGRESO

    st.markdown("#### 🟢 Probabilidad de Empleo Formal")
    st.progress(float(prob_formal))

    st.markdown("#### 🔴 Probabilidad de Empleo Informal")
    st.progress(float(prob_informal))

    st.write("")

    # INTERPRETACIÓN

    st.subheader("🧠 Interpretación del resultado")

    if prediccion == "Formal":

        st.success(
            "El modelo estima que la persona tiene mayor probabilidad de pertenecer al empleo formal."
        )

    else:

        st.warning(
            "El modelo estima que la persona tiene mayor probabilidad de pertenecer al empleo informal."
        )

    st.info(
        "Esta predicción es una estimación estadística basada en el modelo entrenado y no constituye una evaluación laboral oficial."
    )


#------------------------------------------------------------------------------------------------
# PIE DE PÁGINA
#------------------------------------------------------------------------------------------------

st.write("")
st.divider()

st.caption(
    "Aplicación desarrollada con Streamlit | Machine Learning en Producción - Despliegue Web | 2026"
)
