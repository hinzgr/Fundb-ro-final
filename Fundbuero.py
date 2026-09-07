import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import os

# ── Modell laden ───────────────────────────────────────────────────────────────
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ── Datenbank (Liste aller Fundstücke) ────────────────────────────────────────
fundstuecke = []  # Jeder Eintrag: {"name": ..., "beschreibung": ..., "bild_pfad": ...}

# ── KI: Bild erkennen ─────────────────────────────────────────────────────────
def gegenstand_erkennen(bild_pfad):
    bild = Image.open(bild_pfad).convert("RGB")
    bild = bild.resize((224, 224))
    bild_array = np.asarray(bild, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1
    bild_array = np.expand_dims(bild_array, axis=0)
    vorhersage = model.predict(bild_array)
    index = np.argmax(vorhersage)
    return labels[index], vorhersage[0][index]

# ══════════════════════════════════════════════════════════════════════════════
# HAUPT-APP
# ══════════════════════════════════════════════════════════════════════════════
class FundbueroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fundbüro")
        self.geometry("360x640")
        self.resizable(False, False)
        self.configure(bg="white")
        self._zeige_startseite()

    def _alle_widgets_loeschen(self):
        for widget in self.winfo_children():
            widget.destroy()

    # ── STARTSEITE ─────────────────────────────────────────────────────────────
    def _zeige_startseite(self):
        self._alle_widgets_loeschen()
        self.configure(bg="white")

        # Logo oben rechts
        logo_rahmen = tk.Frame(self, bg="white")
        logo_rahmen.pack(anchor="ne", padx=18, pady=(18, 0))
        logo_kreis = tk.Label(logo_rahmen, text="⊕", font=("Arial", 22), bg="white")
        logo_kreis.pack(side="left")
        tu_label = tk.Label(logo_rahmen, text="TU\nES", font=("Arial", 10, "bold"),
                            fg="red", bg="white")
        tu_label.pack(side="left")

        # Abstand
        tk.Label(self, bg="white").pack(pady=60)

        # Button: Suchen
        btn_suchen = tk.Button(
            self,
            text="Suchen  🔍",
            font=("Palatino Linotype", 20, "italic"),
            width=18, height=2,
            relief="solid", bd=2,
            bg="white", fg="black",
            cursor="hand2",
            command=self._zeige_suchseite
        )
        btn_suchen.pack(pady=15)

        # Button: Hochladen
        btn_hochladen = tk.Button(
            self,
            text="Hochladen  ⬆",
            font=("Palatino Linotype", 20, "italic"),
            width=18, height=2,
            relief="solid", bd=2,
            bg="white", fg="black",
            cursor="hand2",
            command=self._zeige_hochladeseite
        )
        btn_hochladen.pack(pady=15)

    # ── SUCHSEITE ──────────────────────────────────────────────────────────────
    def _zeige_suchseite(self):
        self._alle_widgets_loeschen()
        self.configure(bg="white")

        # Suchleiste oben
        such_rahmen = tk.Frame(self, bg="white", relief="solid", bd=2)
        such_rahmen.pack(fill="x", padx=18, pady=(18, 10))

        such_entry = tk.Entry(
            such_rahmen,
            font=("Palatino Linotype", 14, "italic"),
            relief="flat", bd=4,
            fg="gray"
        )
        such_entry.insert(0, "Suchen")
        such_entry.pack(side="left", fill="x", expand=True, padx=5, pady=6)

        such_icon = tk.Label(such_rahmen, text="🔍", font=("Arial", 14), bg="white")
        such_icon.pack(side="right", padx=5)

        # Fokus-Verhalten der Suchleiste
        def eintrag_loeschen(event):
            if such_entry.get() == "Suchen":
                such_entry.delete(0, tk.END)
                such_entry.config(fg="black")

        def suche_aktualisieren(event=None):
            suchbegriff = such_entry.get().lower()
            _liste_aktualisieren(suchbegriff)

        such_entry.bind("<FocusIn>", eintrag_loeschen)
        such_entry.bind("<KeyRelease>", suche_aktualisieren)

        # Scrollbarer Bereich für Fundstücke
        canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="white")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _liste_aktualisieren(suchbegriff=""):
            for widget in scroll_frame.winfo_children():
                widget.destroy()

            gefundene = [f for f in fundstuecke
                         if suchbegriff in f["name"].lower()
                         or suchbegriff in f["beschreibung"].lower()]

            if not gefundene:
                tk.Label(scroll_frame, text="Keine Fundstücke gefunden.",
                         font=("Arial", 12), bg="white", fg="gray").pack(pady=20)
                return

            for fund in gefundene:
                # Karten-Rahmen
                karte = tk.Frame(scroll_frame, bg="white", relief="solid", bd=2)
                karte.pack(fill="x", padx=14, pady=6)

                # Bild links
                try:
                    img = Image.open(fund["bild_pfad"]).resize((70, 70))
                    photo = ImageTk.PhotoImage(img)
                    bild_label = tk.Label(karte, image=photo, bg="white")
                    bild_label.image = photo
                    bild_label.pack(side="left", padx=10, pady=8)
                except Exception:
                    tk.Label(karte, text="🖼", font=("Arial", 30),
                             bg="white").pack(side="left", padx=10, pady=8)

                # Text rechts
                text_rahmen = tk.Frame(karte, bg="white")
                text_rahmen.pack(side="left", anchor="w", pady=6)

                tk.Label(
                    text_rahmen,
                    text=fund["name"],
                    font=("Palatino Linotype", 13, "bold", "underline", "italic"),
                    bg="white"
                ).pack(anchor="w")

                tk.Label(
                    text_rahmen,
                    text="Beschreibung:\n" + fund["beschreibung"],
                    font=("Palatino Linotype", 10, "italic"),
                    bg="white", justify="left"
                ).pack(anchor="w")

        _liste_aktualisieren()

        # Zurück-Button
        tk.Button(
            self, text="← Zurück",
            font=("Arial", 11),
            bg="white", relief="flat",
            cursor="hand2",
            command=self._zeige_startseite
        ).pack(pady=6)

    # ── HOCHLADESEITE ──────────────────────────────────────────────────────────
    def _zeige_hochladeseite(self):
        self._alle_widgets_loeschen()
        self.configure(bg="white")

        # Logo oben rechts
        logo_rahmen = tk.Frame(self, bg="white")
        logo_rahmen.pack(anchor="ne", padx=18, pady=(18, 0))
        tk.Label(logo_rahmen, text="⊕", font=("Arial", 22), bg="white").pack(side="left")
        tk.Label(logo_rahmen, text="TU\nES", font=("Arial", 10, "bold"),
                 fg="red", bg="white").pack(side="left")

        self.foto_pfad = None

        # ── Foto hochladen ──
        def foto_auswaehlen():
            pfad = filedialog.askopenfilename(
                filetypes=[("Bilddateien", "*.jpg *.jpeg *.png *.webp")]
            )
            if pfad:
                self.foto_pfad = pfad
                # KI erkennt den Gegenstand automatisch
                klasse, genauigkeit = gegenstand_erkennen(pfad)
                name_entry.delete(0, tk.END)
                name_entry.insert(0, klasse)
                foto_btn.config(
                    text=f"✅ Foto geladen\n({os.path.basename(pfad)})",
                    fg="green"
                )

        foto_btn = tk.Button(
            self,
            text="Foto hochladen  ⬆",
            font=("Palatino Linotype", 16, "italic"),
            width=22, height=2,
            relief="solid", bd=2,
            bg="white", fg="black",
            cursor="hand2",
            command=foto_auswaehlen
        )
        foto_btn.pack(pady=(30, 12))

        # ── Name des Objekts ──
        name_entry = tk.Entry(
            self,
            font=("Palatino Linotype", 16, "italic"),
            relief="solid", bd=2,
            width=22,
            fg="gray"
        )
        name_entry.insert(0, "Name des Objekts")
        name_entry.pack(ipady=14, pady=12)

        def name_fokus(event):
            if name_entry.get() == "Name des Objekts":
                name_entry.delete(0, tk.END)
                name_entry.config(fg="black")

        name_entry.bind("<FocusIn>", name_fokus)

        # ── Beschreibung ──
        beschreibung_entry = tk.Entry(
            self,
            font=("Palatino Linotype", 16, "italic"),
            relief="solid", bd=2,
            width=22,
            fg="gray"
        )
        beschreibung_entry.insert(0, "Beschreibung")
        beschreibung_entry.pack(ipady=14, pady=12)

        def beschr_fokus(event):
            if beschreibung_entry.get() == "Beschreibung":
                beschreibung_entry.delete(0, tk.END)
                beschreibung_entry.config(fg="black")

        beschreibung_entry.bind("<FocusIn>", beschr_fokus)

        # ── Fertig-Button ──
        def fertig():
            name = name_entry.get()
            beschreibung = beschreibung_entry.get()

            if not self.foto_pfad:
                messagebox.showwarning("Kein Foto", "Bitte lade zuerst ein Foto hoch!")
                return
            if name == "" or name == "Name des Objekts":
                messagebox.showwarning("Kein Name", "Bitte gib einen Namen ein!")
                return
            if beschreibung == "" or beschreibung == "Beschreibung":
                beschreibung = "–"

            fundstuecke.append({
                "name": name,
                "beschreibung": beschreibung,
                "bild_pfad": self.foto_pfad
            })
            messagebox.showinfo("Gespeichert", f"'{name}' wurde im Fundbüro gespeichert! ✅")
            self._zeige_startseite()

        tk.Button(
            self,
            text="Fertig",
            font=("Palatino Linotype", 14, "italic"),
            bg="#3dd6b5", fg="white",
            relief="flat", bd=0,
            width=14, height=2,
            cursor="hand2",
            command=fertig
        ).pack(pady=18)

        # Zurück
        tk.Button(
            self, text="← Zurück",
            font=("Arial", 11),
            bg="white", relief="flat",
            cursor="hand2",
            command=self._zeige_startseite
        ).pack()

# ── App starten ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FundbueroApp()
    app.mainloop()
