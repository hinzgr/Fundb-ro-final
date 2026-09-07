import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# ── Seiteneinstellungen ────────────────────────────────────────────────────────
st.set_page_config(page_title="Fundbüro", page_icon="🎒", layout="centered")

# ── CSS: Design anpassen ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hintergrund weiß */
    .stApp { background-color: white; }

    /* Große kursive Schrift für Buttons */
    .grosser-button > button {
        font-family: 'Palatino Linotype', serif !important;
        font-size: 22px !important;
        font-style: italic !important;
        width: 100% !important;
        height: 80px !important;
        background-color: white !important;
        color: black !important;
        border: 2px solid black !important;
        border-radius: 6px !important;
        margin-bottom: 16px !important;
    }
    .grosser-button > button:hover {
        background-color: #f0f0f0 !important;
    }

    /* Fertig-Button türkis */
    .fertig-button > button {
        font-family: 'Palatino Linotype', serif !important;
        font-size: 18px !important;
        font-style: italic !important;
        background-color: #3dd6b5 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        width: 100% !important;
        height: 55px !important;
    }

    /* Logo oben rechts */
    .logo {
        text-align: right;
        font-size: 14px;
        font-weight: bold;
        color: red;
        margin-bottom: 40px;
    }

    /* Fundstück-Karte */
    .karte {
        display: flex;
        align-items: center;
        border: 2px solid black;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 12px;
        background-color: white;
    }
    .karte-text {
        margin-left: 16px;
    }
    .karte-name {
        font-family: 'Palatino Linotype', serif;
        font-size: 17px;
        font-style: italic;
        font-weight: bold;
        text-decoration: underline;
    }
    .karte-beschreibung {
        font-family: 'Palatino Linotype', serif;
        font-size: 13px;
        font-style: italic;
        color: #333;
    }

    /* Suchfeld */
    .stTextInput > div > div > input {
        font-family: 'Palatino Linotype', serif !important;
        font-style: italic !important;
        font-size: 16px !important;
        border: 2px solid black !important;
        border-radius: 6px !important;
    }

    /* Trennlinien entfernen */
    hr { display: none; }

    /* Streamlit-Header ausblenden */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Modell laden ───────────────────────────────────────────────────────────────
@st.cache_resource
def modell_laden():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = modell_laden()

# ── KI: Bild erkennen ─────────────────────────────────────────────────────────
def gegenstand_erkennen(bild: Image.Image):
    bild = bild.convert("RGB").resize((224, 224))
    bild_array = np.asarray(bild, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1
    bild_array = np.expand_dims(bild_array, axis=0)
    vorhersage = model.predict(bild_array)
    index = np.argmax(vorhersage)
    return labels[index], float(vorhersage[0][index])

# ── Session State initialisieren ───────────────────────────────────────────────
if "seite" not in st.session_state:
    st.session_state.seite = "start"
if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = []

# ══════════════════════════════════════════════════════════════════════════════
# STARTSEITE
# ══════════════════════════════════════════════════════════════════════════════
def startseite():
    # Logo oben rechts
    st.markdown('<div class="logo">⊕ TU<br>ES</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Button: Suchen
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown('<div class="grosser-button">', unsafe_allow_html=True)
        if st.button("Suchen  🔍"):
            st.session_state.seite = "suchen"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="grosser-button">', unsafe_allow_html=True)
        if st.button("Hochladen  ⬆"):
            st.session_state.seite = "hochladen"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SUCHSEITE
# ══════════════════════════════════════════════════════════════════════════════
def suchseite():
    # Suchfeld
    suchbegriff = st.text_input("", placeholder="Suchen 🔍")

    st.markdown("<br>", unsafe_allow_html=True)

    # Fundstücke filtern
    gefundene = [
        f for f in st.session_state.fundstuecke
        if suchbegriff.lower() in f["name"].lower()
        or suchbegriff.lower() in f["beschreibung"].lower()
    ]

    if not gefundene:
        st.markdown(
            "<p style='font-family:Palatino Linotype;font-style:italic;"
            "color:gray;text-align:center;'>Keine Fundstücke gefunden.</p>",
            unsafe_allow_html=True
        )
    else:
        for fund in gefundene:
            col_bild, col_text = st.columns([1, 3])
            with col_bild:
                if fund["bild"] is not None:
                    st.image(fund["bild"], width=70)
                else:
                    st.markdown("🖼", unsafe_allow_html=True)
            with col_text:
                st.markdown(
                    f'<div class="karte-name">{fund["name"]}</div>'
                    f'<div class="karte-beschreibung">Beschreibung:<br>{fund["beschreibung"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<hr style='border:1px solid black;display:block;'>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Zurück"):
        st.session_state.seite = "start"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HOCHLADESEITE
# ══════════════════════════════════════════════════════════════════════════════
def hochladeseite():
    # Logo oben rechts
    st.markdown('<div class="logo">⊕ TU<br>ES</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        # ── Foto hochladen ──
        hochgeladenes_foto = st.file_uploader(
            "Foto hochladen ⬆",
            type=["jpg", "jpeg", "png", "webp"]
        )

        erkannter_name = ""

        if hochgeladenes_foto is not None:
            bild = Image.open(hochgeladenes_foto)
            st.image(bild, width=120)

            # KI erkennt automatisch
            with st.spinner("KI erkennt Gegenstand..."):
                klasse, genauigkeit = gegenstand_erkennen(bild)
            erkannter_name = klasse
            st.success(f"KI erkannt: **{klasse}** ({genauigkeit*100:.1f}% sicher)")

        # ── Name des Objekts ──
        name = st.text_input(
            "",
            value=erkannter_name,
            placeholder="Name des Objekts"
        )

        # ── Beschreibung ──
        beschreibung = st.text_input("", placeholder="Beschreibung")

        # ── Fertig-Button ──
        st.markdown('<div class="fertig-button">', unsafe_allow_html=True)
        fertig = st.button("Fertig")
        st.markdown('</div>', unsafe_allow_html=True)

        if fertig:
            if hochgeladenes_foto is None:
                st.warning("⚠️ Bitte lade zuerst ein Foto hoch!")
            elif name.strip() == "":
                st.warning("⚠️ Bitte gib einen Namen ein!")
            else:
                bild_gespeichert = Image.open(hochgeladenes_foto)
                st.session_state.fundstuecke.append({
                    "name": name,
                    "beschreibung": beschreibung if beschreibung.strip() != "" else "–",
                    "bild": bild_gespeichert
                })
                st.success(f"✅ '{name}' wurde im Fundbüro gespeichert!")
                st.session_state.seite = "start"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Zurück"):
        st.session_state.seite = "start"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SEITENSTEUERUNG
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.seite == "start":
    startseite()
elif st.session_state.seite == "suchen":
    suchseite()
elif st.session_state.seite == "hochladen":
    hochladeseite()
