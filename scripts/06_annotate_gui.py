#!/usr/bin/env python3
"""
Manuelle Evaluations-GUI für FVG-Annotation.

Zeigt Sätze aus eval_sentences.csv nacheinander an.
Verb und Nomen sind farblich hervorgehoben.
Tastenkürzel: J = Ja, N = Nein, U = Unklar, ← = Zurück
"""

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_sentences.csv"

COLOR_VERB   = "#FF8C00"   # Orange – Funktionsverb
COLOR_NOM    = "#1E90FF"   # Blau   – Nomen
COLOR_BG     = "#F8F8F8"
COLOR_PANEL  = "#EFEFEF"
COLOR_JA     = "#2ECC71"
COLOR_NEIN   = "#E74C3C"
COLOR_UNKLAR = "#F39C12"
COLOR_BACK   = "#95A5A6"


# ── CSV laden / speichern ──────────────────────────────────────────────────────

def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


# ── GUI ───────────────────────────────────────────────────────────────────────

class AnnotationApp:
    def __init__(self, root: tk.Tk, rows: list[dict], fieldnames: list[str]):
        self.root       = root
        self.rows       = rows
        self.fieldnames = fieldnames
        self.index      = self._find_first_unannotated()

        root.title("FVG-Annotation")
        root.configure(bg=COLOR_BG)
        root.geometry("860x560")
        root.minsize(700, 460)
        root.bind("<Key>", self._on_key)

        self._build_ui()
        self._show_current()

    def _find_first_unannotated(self) -> int:
        for i, r in enumerate(self.rows):
            if r.get("manuell_FVG", "").strip() == "":
                return i
        return len(self.rows)   # alles fertig

    # ── UI-Aufbau ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Fortschrittsbalken oben ────────────────────────────────────────────
        top = tk.Frame(root, bg=COLOR_BG, pady=6, padx=14)
        top.pack(fill="x")

        self.lbl_progress = tk.Label(
            top, text="", bg=COLOR_BG, fg="#555",
            font=("Helvetica", 10),
        )
        self.lbl_progress.pack(side="left")

        self.canvas_bar = tk.Canvas(top, height=8, bg="#DDD", highlightthickness=0)
        self.canvas_bar.pack(side="left", fill="x", expand=True, padx=(10, 0))

        # ── Paar-Info-Panel ────────────────────────────────────────────────────
        panel = tk.Frame(root, bg=COLOR_PANEL, padx=14, pady=8)
        panel.pack(fill="x", padx=14)

        self.lbl_pair = tk.Label(
            panel, text="", bg=COLOR_PANEL, fg="#222",
            font=("Helvetica", 12, "bold"), anchor="w", justify="left",
        )
        self.lbl_pair.pack(fill="x")

        meta_row = tk.Frame(panel, bg=COLOR_PANEL)
        meta_row.pack(fill="x", pady=(2, 0))
        self.lbl_csv   = tk.Label(meta_row, text="", bg=COLOR_PANEL, fg="#444",
                                  font=("Helvetica", 10), anchor="w")
        self.lbl_csv.pack(side="left", padx=(0, 20))
        self.lbl_ds    = tk.Label(meta_row, text="", bg=COLOR_PANEL, fg="#444",
                                  font=("Helvetica", 10), anchor="w")
        self.lbl_ds.pack(side="left")

        # ── Satz-Anzeige ───────────────────────────────────────────────────────
        sent_frame = tk.Frame(root, bg=COLOR_BG, padx=14, pady=10)
        sent_frame.pack(fill="both", expand=True)

        serif = tkfont.Font(family="Georgia", size=13)
        self.txt = tk.Text(
            sent_frame, wrap="word", font=serif,
            bg="#FFFFFF", relief="flat", bd=0,
            padx=10, pady=10, cursor="arrow",
            state="disabled", height=8,
        )
        self.txt.pack(fill="both", expand=True)

        self.txt.tag_config("verb",    background=COLOR_VERB,   foreground="#000")
        self.txt.tag_config("nom",     background=COLOR_NOM,    foreground="#000")
        self.txt.tag_config("plain",   foreground="#222")
        self.txt.tag_config("current", background="#FFFDE7")    # vorher annotiert

        # ── Legende ────────────────────────────────────────────────────────────
        leg = tk.Frame(root, bg=COLOR_BG, padx=14)
        leg.pack(fill="x")
        tk.Label(leg, text="■ Verb", bg=COLOR_VERB,  fg="#000",
                 font=("Helvetica", 9), padx=4).pack(side="left", padx=(0, 6))
        tk.Label(leg, text="■ Nomen", bg=COLOR_NOM, fg="#000",
                 font=("Helvetica", 9), padx=4).pack(side="left")

        # ── Knöpfe ─────────────────────────────────────────────────────────────
        btn_row = tk.Frame(root, bg=COLOR_BG, pady=14, padx=14)
        btn_row.pack(fill="x")

        btn_cfg = dict(font=("Helvetica", 14, "bold"), width=8,
                       relief="flat", cursor="hand2", padx=4, pady=8)

        self.btn_ja = tk.Button(
            btn_row, text="Ja  [J]", bg=COLOR_JA, fg="#FFF",
            command=lambda: self._annotate("1"), **btn_cfg,
        )
        self.btn_ja.pack(side="left", padx=(0, 8))

        self.btn_nein = tk.Button(
            btn_row, text="Nein  [N]", bg=COLOR_NEIN, fg="#FFF",
            command=lambda: self._annotate("0"), **btn_cfg,
        )
        self.btn_nein.pack(side="left", padx=(0, 8))

        self.btn_unklar = tk.Button(
            btn_row, text="Unklar  [U]", bg=COLOR_UNKLAR, fg="#FFF",
            command=lambda: self._annotate("?"), **btn_cfg,
        )
        self.btn_unklar.pack(side="left", padx=(0, 20))

        self.btn_back = tk.Button(
            btn_row, text="← Zurück  [←]", bg=COLOR_BACK, fg="#FFF",
            command=self._go_back, **btn_cfg,
        )
        self.btn_back.pack(side="left")

        # Aktuelles Label (rechts)
        self.lbl_current = tk.Label(
            btn_row, text="", bg=COLOR_BG, fg="#888",
            font=("Helvetica", 10), anchor="e",
        )
        self.lbl_current.pack(side="right", padx=4)

    # ── Anzeige aktualisieren ──────────────────────────────────────────────────

    def _show_current(self):
        n = len(self.rows)

        # Fertig?
        if self.index >= n:
            self._show_done()
            return

        row = self.rows[self.index]

        # Fortschritt
        done = sum(1 for r in self.rows if r.get("manuell_FVG", "").strip() != "")
        pct  = done / n if n else 0
        self.lbl_progress.config(text=f"Satz {self.index + 1} / {n}  ({done} annotiert)")
        self.canvas_bar.update_idletasks()
        w = self.canvas_bar.winfo_width()
        self.canvas_bar.delete("all")
        self.canvas_bar.create_rectangle(0, 0, int(w * pct), 8, fill="#2ECC71", outline="")

        # Paar-Info
        verb  = row.get("verb_lemma", "")
        noun  = row.get("noun_lemma", "")
        full  = row.get("full_pattern", "").strip()
        label = full if full else f"{noun}  +  {verb}"
        self.lbl_pair.config(text=label)

        csv_lbl = row.get("fvg_csv", "").strip()
        ds_lbl  = row.get("distant_label", "").strip()
        if csv_lbl == "1":
            csv_txt = "CSV-Label: FVG"
        elif csv_lbl == "0":
            csv_txt = "CSV-Label: kein FVG"
        else:
            csv_txt = "CSV-Label: (kein Vorab-Label)"
        ds_txt = f"Distant Supervision: {ds_lbl}" if ds_lbl else "Distant Supervision: –"
        self.lbl_csv.config(text=csv_txt)
        self.lbl_ds.config(text=ds_txt)

        # Satz mit Highlighting
        sentence  = row.get("sentence", "")
        verb_tok  = row.get("verb_token", "")
        noun_tok  = row.get("noun_token", "")
        praep_tok = row.get("praep", "").strip()
        self._render_sentence(sentence, verb_tok, noun_tok, praep_tok)

        # Aktuelles Annotation-Label
        curr = row.get("manuell_FVG", "").strip()
        if curr:
            label_map = {"1": "Ja", "0": "Nein", "?": "Unklar"}
            self.lbl_current.config(text=f"Aktuelle Markierung: {label_map.get(curr, curr)}")
        else:
            self.lbl_current.config(text="")

        # Zurück-Button
        self.btn_back.config(state="normal" if self.index > 0 else "disabled")

    def _render_sentence(self, sentence: str, verb_tok: str, noun_tok: str,
                         praep_tok: str = ""):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")

        words  = sentence.split(" ")
        output = []   # list of (text, tag)

        # Für Präp-FVG: markiere die Präposition ebenfalls als NOM (blau)
        # Nur das erste Vorkommen der Präp direkt vor dem Nomen markieren
        noun_positions = [i for i, w in enumerate(words)
                          if w.rstrip(".,;:!?\"'()") == noun_tok and noun_tok]

        praep_to_mark: set[int] = set()
        if praep_tok:
            for ni in noun_positions:
                if ni > 0:
                    candidate = words[ni - 1].rstrip(".,;:!?\"'()")
                    if candidate.lower() == praep_tok.lower():
                        praep_to_mark.add(ni - 1)

        for i, word in enumerate(words):
            bare = word.rstrip(".,;:!?\"'()")
            if bare == verb_tok and verb_tok:
                output.append((word + " ", "verb"))
            elif i in praep_to_mark:
                output.append((word + " ", "nom"))
            elif bare == noun_tok and noun_tok:
                output.append((word + " ", "nom"))
            else:
                output.append((word + " ", "plain"))

        for text, tag in output:
            self.txt.insert("end", text, tag)

        self.txt.config(state="disabled")

    def _show_done(self):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "\n\n  Alle Sätze wurden annotiert. Danke!\n\n"
                        "  Die Ergebnisse wurden gespeichert in:\n"
                        f"  {CSV_FILE}", "plain")
        self.txt.config(state="disabled")
        self.lbl_pair.config(text="Annotation abgeschlossen ✓")
        self.lbl_csv.config(text="")
        self.lbl_ds.config(text="")
        self.lbl_current.config(text="")
        for btn in (self.btn_ja, self.btn_nein, self.btn_unklar):
            btn.config(state="disabled")

    # ── Aktionen ───────────────────────────────────────────────────────────────

    def _annotate(self, value: str):
        if self.index >= len(self.rows):
            return
        self.rows[self.index]["manuell_FVG"] = value
        save_csv(CSV_FILE, self.rows, self.fieldnames)
        self.index += 1
        # Überspringe bereits annotierte Sätze vorwärts (für Resume)
        while (self.index < len(self.rows)
               and self.rows[self.index].get("manuell_FVG", "").strip() != ""):
            self.index += 1
        self._show_current()

    def _go_back(self):
        if self.index <= 0:
            return
        self.index -= 1
        # Gehe zum letzten annotierten Satz
        while (self.index > 0
               and self.rows[self.index].get("manuell_FVG", "").strip() == ""):
            self.index -= 1
        self._show_current()

    def _on_key(self, event):
        key = event.keysym.lower()
        if key == "j":
            self._annotate("1")
        elif key == "n":
            self._annotate("0")
        elif key in ("u", "question"):
            self._annotate("?")
        elif key == "left":
            self._go_back()


# ── Einstiegspunkt ─────────────────────────────────────────────────────────────

def main():
    if not CSV_FILE.exists():
        print(f"Fehler: {CSV_FILE} nicht gefunden.")
        print("Zuerst 05_extract_eval_sentences.py ausführen.")
        sys.exit(1)

    rows, fieldnames = load_csv(CSV_FILE)
    if not rows:
        print("CSV ist leer.")
        sys.exit(1)

    done  = sum(1 for r in rows if r.get("manuell_FVG", "").strip() != "")
    total = len(rows)
    print(f"Lade {total} Sätze ({done} bereits annotiert) …")

    root = tk.Tk()
    app  = AnnotationApp(root, rows, fieldnames)
    root.mainloop()


if __name__ == "__main__":
    main()
