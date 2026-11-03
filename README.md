# Duotrigordle Solver

Ein heuristischer Solver für [Duotrigordle](https://duotrigordle.com/), also 32 Wordle-Boards gleichzeitig mit 37 Versuchen. Ein Violentmonkey-Userscript verbindet ihn mit der echten Webseite.

Der Solver ist auf **möglichst wenige Versuche** ausgelegt, nicht auf Geschwindigkeit. In jedem Zug wird jedes erlaubte Wort gegen alle offenen Boards bewertet.

```
Browser (duotrigordle.com)                         Python (lokal)
┌──────────────────────────────┐   POST /next    ┌──────────────────────────────┐
│ duo-bridge.user.js           │ ──────────────▶ │ duo_bridge_server.py         │
│ • liest 32 Boards aus dem DOM│   Spielverlauf  │ • rekonstruiert Kandidaten   │
│ • tippt den Guess ein        │ ◀────────────── │ • duotrigordle_solver.py     │
└──────────────────────────────┘   nächster Guess│   wählt den nächsten Guess   │
                                                 └──────────────────────────────┘
```
 
 
 
## Funktionen

- Exakte Wordle-Logik inklusive doppelter Buchstaben
- Bewertung über erwartete Restunsicherheit, Informationsgewinn und Kandidatenwahrscheinlichkeiten
- Bereits feststehende Boards (nur noch ein Kandidat) werden bevorzugt gelöst
- Rechnet mit NumPy auf der CPU (Multiprocessing) oder optional mit CuPy auf der GPU
- Kann mitten im Spiel übernehmen, auch nach eigenen, manuellen Eingaben
- Fehlt ein Zielwort in der Lösungsliste, wird nur dieses Board automatisch auf alle gültigen Wörter erweitert
- Gelöste Zielwörter, die unbekannt waren, werden gelernt und kommen in künftigen Spielen mit
- Jede Entscheidung wird als JSONL protokolliert, praktisch für spätere Auswertungen
 
 
 
## Voraussetzungen

- Python 3.9 oder neuer
- `numpy` (Pflicht), `wordfreq` (empfohlen, verbessert die Priors), `cupy-cudaXXx` (optional für GPU)
- Firefox oder ein Chromium-Browser mit [Violentmonkey](https://violentmonkey.github.io/)
- Linux wird empfohlen (dort wird ein Prozess-Pool genutzt). Andere Systeme laufen mit Threads.
 
 
 
## Installation

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# optional für NVIDIA-GPUs, passend zur CUDA-Version:
# pip install cupy-cuda12x
```
 

  
### Userscript installieren

Wähle einen der Wege:

1. **Über den laufenden Server:** `python3 duo_bridge_server.py` starten und `http://127.0.0.1:8765/duo-bridge.user.js` im Browser öffnen. Violentmonkey bietet dann die Installation an.
2. **Direkt von GitHub:** `https://raw.githubusercontent.com/<USER>/<REPO>/main/duo-bridge.user.js` öffnen.
3. **Manuell:** Im Violentmonkey-Dashboard auf „+“ klicken, ein neues Script anlegen und den Inhalt von `duo-bridge.user.js` einfügen.
 
 
 
## Benutzung

1. Server starten:
   ```bash
   python3 duo_bridge_server.py              # Auto: GPU wenn CuPy vorhanden, sonst CPU
   python3 duo_bridge_server.py --backend cpu
   ```
2. [duotrigordle.com](https://duotrigordle.com/) öffnen und ein normales Spiel starten (Daily oder Practice).
3. Unten rechts erscheint das Panel **Solver Bridge**:
   - **▶ Start** spielt, bis das Spiel vorbei ist
   - **⏭ Schritt** spielt genau einen Zug
   - **■ Stop** hält nach dem laufenden Zug an
 
 
 
### Server-Optionen

| Option | Bedeutung |
|---|---|
| `--backend auto\|cpu\|gpu` | Rechenbackend (Standard: `auto`) |
| `--port 8765` | Port. Muss zu `SERVER` im Userscript passen. |
| `--batch-size N` | Guess-Wörter pro Batch. Auf der GPU sind größere Werte (z. B. 4000) oft schneller. |
| `--wide-solutions` | Alle 14.855 gültigen Wörter von Anfang an als mögliche Lösungen. Sicherer, aber langsamer. |
| `--no-cache` | Zwischengespeicherte Entscheidungen ignorieren |
| `--quiet` | Weniger Ausgaben |
| `--solutions`, `--guesses`, `--solver` | Eigene Wortlisten oder Solver-Datei verwenden |



### Ausgabedateien (`output/`)

| Datei | Inhalt |
|---|---|
| `live_games.jsonl` | Ein Eintrag pro Spiel: gelöste Boards, Zug je Board, gespielte Wörter |
| `live_decisions.jsonl` | Jede Entscheidung mit allen Kennzahlen |
| `solutions_learned.txt` | Zielwörter, die nicht in der Lösungsliste standen |
| `bridge_cache.json` | Cache der Entscheidungen. Der erste Zug wird so nur einmal berechnet. |
 
 
 
## Laufzeit

Der erste Zug ist der teuerste, weil jedes Wort gegen alle Kandidaten auf allen 32 Boards bewertet wird. Mit `--wide-solutions` sind das rund 7 Milliarden Paare, auf einer GPU etwa 5 Minuten. Danach landet die Entscheidung im Cache. Die späteren Züge werden mit schrumpfenden Kandidatenmengen schneller.
 
 
 
## Solver ohne Browser

`duotrigordle_solver.py` lässt sich auch allein starten. Er simuliert dann Spiele mit zufälligen Zielwörtern aus `data/solutions_en.txt`:

```bash
python3 duotrigordle_solver.py
```
 
 
 
## Fehlersuche

| Problem | Lösung |
|---|---|
| Panel meldet „Server nicht erreichbar“ | Läuft `duo_bridge_server.py`? Stimmen Port und `SERVER` im Userscript überein? |
| „X Boards gefunden statt 32“ oder „verschiedene Wörter“ | Normales Spiel öffnen. Der Sequence-Modus mit verdeckten Boards wird nicht unterstützt. |
| „kein gültiges Wort passt …“ | Die Farben wurden falsch gelesen. In der Browser-Konsole `DuoBridge.dump()` ausführen und die Zellklassen prüfen. Die Seite kann ihre CSS-Klassen geändert haben, dann `COLOR_CLASS` im Userscript anpassen. |
| Es wird nichts getippt | Das Script wechselt automatisch auf Klicks der Bildschirmtastatur. Erzwingen lässt es sich mit `INPUT_METHOD: "click"`. |
| Wort wird von der Seite abgelehnt | Es wird automatisch ausgeschlossen. Zurücksetzen über das Violentmonkey-Menü „Ausschlussliste leeren“. |

  
 
 
## Wortlisten

`data/solutions_en.txt` (2.313 Lösungen) und `data/guesses_en.txt` (14.855 gültige Wörter) stammen aus dem früher quelloffenen Frontend von Duotrigordle, Copyright © 2022 Bryan Chen, MIT-Lizenz (siehe [`data/LICENSE-wordlists`](data/LICENSE-wordlists)). Die Live-Seite kann inzwischen andere Listen verwenden. Dafür gibt es den automatischen Fallback und die gelernten Wörter.
