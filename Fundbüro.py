import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ── Modell laden ──────────────────────────────────────────────────────────────
MODEL_PATH = "keras_model.h5"       # Pfad zu deinem Teachable Machine Modell
LABELS_PATH = "labels.txt"          # Pfad zu deiner Labels-Datei

model = load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ── Fundbüro-Datenbank ────────────────────────────────────────────────────────
fundstuecke = []  # Liste aller gefundenen Gegenstände

def gegenstand_erkennen(bild):
    """Verarbeitet ein Bild und gibt die erkannte Klasse zurück."""
    bild_resized = cv2.resize(bild, (224, 224))
    bild_array = np.asarray(bild_resized, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1  # Normalisierung wie bei Teachable Machine
    bild_array = np.expand_dims(bild_array, axis=0)

    vorhersage = model.predict(bild_array)
    index = np.argmax(vorhersage)
    klasse = labels[index]
    genauigkeit = vorhersage[0][index]
    return klasse, genauigkeit

def gegenstand_einlagern(klasse, genauigkeit):
    """Speichert einen erkannten Gegenstand in der Datenbank."""
    nummer = len(fundstuecke) + 1
    eintrag = {
        "nummer": nummer,
        "gegenstand": klasse,
        "genauigkeit": f"{genauigkeit * 100:.1f}%"
    }
    fundstuecke.append(eintrag)
    print(f"✅ Gegenstand #{nummer} eingelagert: {klasse} ({eintrag['genauigkeit']} sicher)")

def alle_fundstuecke_anzeigen():
    """Zeigt alle eingelagerten Fundstücke an."""
    if not fundstuecke:
        print("📭 Keine Fundstücke vorhanden.")
        return
    print("\n📦 Fundstücke im Lager:")
    print("-" * 40)
    for eintrag in fundstuecke:
        print(f"  #{eintrag['nummer']} | {eintrag['gegenstand']} | Genauigkeit: {eintrag['genauigkeit']}")
    print("-" * 40)

# ── Kamera & Hauptprogramm ────────────────────────────────────────────────────
kamera = cv2.VideoCapture(0)  # 0 = Standard-Webcam

print("🎥 Kamera gestartet.")
print("  [LEERTASTE] → Gegenstand erkennen & einlagern")
print("  [L]         → Alle Fundstücke anzeigen")
print("  [Q]         → Beenden\n")

while True:
    ret, frame = kamera.read()
    if not ret:
        print("❌ Kamera nicht gefunden!")
        break

    # Aktuell erkannte Klasse live anzeigen
    klasse, genauigkeit = gegenstand_erkennen(frame)
    anzeige_text = f"{klasse} ({genauigkeit * 100:.1f}%)"
    cv2.putText(frame, anzeige_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.imshow("Fundbüro - Kamera", frame)

    taste = cv2.waitKey(1) & 0xFF

    if taste == ord('q'):
        print("👋 Programm beendet.")
        break
    elif taste == ord(' '):
        gegenstand_einlagern(klasse, genauigkeit)
    elif taste == ord('l'):
        alle_fundstuecke_anzeigen()

kamera.release()
cv2.destroyAllWindows()
