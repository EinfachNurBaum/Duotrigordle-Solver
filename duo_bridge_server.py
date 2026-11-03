#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duotrigordle Bridge Server

Lokaler HTTP-Server zwischen dem Userscript im Browser und duotrigordle_solver.py.
Der Browser schickt den sichtbaren Spielverlauf, der Server liefert den nächsten Guess.
Der Server ist zustandslos: Er rekonstruiert alle Boards bei jeder Anfrage neu.

    python3 duo_bridge_server.py [--backend gpu] [--port 8765] [--wide-solutions]
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"

FEEDBACK_CHARS = {"G": 2, "Y": 1, ".": 0, "B": 0, "-": 0}


class BridgeError(Exception):
    """Fehler, der als HTTP 422 an das Userscript zurückgeht."""


def load_solver(path: Path):
    """Solver-Modul aus einer Datei laden."""
    spec = importlib.util.spec_from_file_location("duotrigordle_solver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["duotrigordle_solver"] = module
    spec.loader.exec_module(module)
    return module


def read_words(path: Path):
    """Wortmenge aus einer Textdatei lesen (leer, falls nicht vorhanden)."""
    if not path.exists():
        return set()
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").split() if w.strip()}


class Bridge:
    """Hält Wortdaten und Solver und beantwortet /next und /finish."""

    def __init__(self, args):
        """Solver konfigurieren, Wortlisten laden und Cache öffnen."""
        self.args = args
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.solver = s = load_solver(Path(args.solver))

        self.learned_path = OUTPUT_DIR / "solutions_learned.txt"
        if args.wide_solutions:
            solution_file = Path(args.guesses)
        else:
            merged = read_words(Path(args.solutions)) | read_words(self.learned_path)
            solution_file = OUTPUT_DIR / "solutions_merged.txt"
            solution_file.write_text("\n".join(sorted(merged)) + "\n", encoding="utf-8")

        s.DB_FILE = None
        s.TEXT_FILE = args.guesses
        s.GUESS_TEXT_FILE = args.guesses
        s.SOLUTION_TEXT_FILE = solution_file
        s.WORDFREQ_LANGUAGE = "en"
        s.COMPUTATION_BACKEND = args.backend
        s.PRINT_PROGRESS_EVERY_BATCH = not args.quiet
        s.DECISION_LOG_FILE = OUTPUT_DIR / "live_decisions.jsonl"
        s.SINGLETON_LOG_FILE = OUTPUT_DIR / "live_singletons.csv"
        if args.batch_size:
            s.GUESS_BATCH_SIZE = args.batch_size

        sol, sol_counts, guess, guess_counts = s.load_word_data()
        self.wd = s.WordData(sol, sol_counts, guess, guess_counts)
        self.backend = s.choose_backend()
        self.num_boards = s.NUM_BOARDS
        self.base_solutions = set(self.wd.solution_words)
        self.all_valid = set(self.wd.guess_words) | self.base_solutions
        self.widened = []

        self.cache_path = OUTPUT_DIR / "bridge_cache.json"
        self.cache = {}
        if not args.no_cache and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.cache = {}

        h = hashlib.sha256()
        for p in (args.solver, args.guesses, solution_file):
            h.update(Path(p).read_bytes())
        h.update(repr((s.WEIGHTS, s.WORDFREQ_LANGUAGE, s.zipf_frequency is not None)).encode())
        self.fingerprint = h.hexdigest()[:16]

        print(f"\nBridge bereit | Backend={self.backend} | Lösungen={len(sol)} | "
              f"Guesses={len(guess)} | Fingerprint={self.fingerprint}")

    def parse_feedback(self, text, board, guess_no):
        """Feedback-String wie 'GY..G' in ein Tupel (2,1,0,0,2) umwandeln."""
        text = str(text).strip().upper()
        if len(text) != 5 or any(c not in FEEDBACK_CHARS for c in text):
            raise BridgeError(f"Ungültiges Feedback '{text}' auf Board {board + 1}, "
                              f"Guess {guess_no}")
        return tuple(FEEDBACK_CHARS[c] for c in text)

    def reconstruct(self, history):
        """Kandidatenmengen aus dem Verlauf aufbauen; leere Boards auf alle gültigen Wörter erweitern."""
        s = self.solver
        n = self.num_boards
        cand = [set(self.base_solutions) for _ in range(n)]
        solved = [False] * n
        solved_at = [None] * n
        widened = [False] * n
        parsed = []

        for k, entry in enumerate(history, start=1):
            guess = str(entry["guess"]).strip().lower()
            fbs = entry["feedback"]
            if len(guess) != 5 or not guess.isalpha():
                raise BridgeError(f"Ungültiger Guess '{guess}' in Zeile {k}")
            if len(fbs) != n:
                raise BridgeError(f"Zeile {k}: {len(fbs)} Feedbacks statt {n}")
            parsed.append((guess, [None if f is None else self.parse_feedback(f, b, k)
                                   for b, f in enumerate(fbs)]))

            for b in range(n):
                if solved[b]:
                    continue
                fb = parsed[-1][1][b]
                if fb is None:
                    raise BridgeError(f"Board {b + 1} ist ungelöst, hat aber kein Feedback "
                                      f"für Guess {k} ({guess.upper()}).")
                if fb == (2, 2, 2, 2, 2):
                    solved[b], solved_at[b], cand[b] = True, k, {guess}
                    continue
                new = s.filter_candidates(cand[b], guess, fb)
                if not new and not widened[b]:
                    widened[b] = True
                    new = set(self.all_valid)
                    for g, row in parsed:
                        new = s.filter_candidates(new, g, row[b])
                if not new:
                    raise BridgeError(
                        f"Board {b + 1}: kein gültiges Wort passt nach Guess {k} "
                        f"({guess.upper()} -> {fbs[b]}). Gelesenes Feedback prüfen "
                        f"(DuoBridge.dump() in der Browser-Konsole).")
                cand[b] = new

        self.widened = [b + 1 for b in range(n) if widened[b]]
        return cand, solved, solved_at

    def next_guess(self, payload):
        """Nächsten Guess für den übermittelten Spielstand berechnen (mit Cache)."""
        s, wd = self.solver, self.wd
        history = payload.get("history", [])
        exclude = sorted({str(w).lower() for w in payload.get("exclude", [])})
        game_id = payload.get("game_id") or datetime.now().strftime("%Y-%m-%d")

        cand, solved, solved_at = self.reconstruct(history)
        active = [b for b in range(self.num_boards) if not solved[b]]
        if not active:
            return {"done": True, "solved_at": solved_at}

        key = hashlib.sha256(json.dumps([self.fingerprint, history, exclude],
                                        sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            print(f"[Cache] Guess {len(history) + 1}: {self.cache[key]['guess'].upper()}")
            return dict(self.cache[key], cached=True)

        if self.widened:
            print(f"[Info] Boards {self.widened} auf alle gültigen Wörter erweitert.")

        idx_by_board = [np.array(sorted(wd.word_to_index[w] for w in cand[b]
                                        if w in wd.word_to_index), dtype=np.int32)
                        for b in range(self.num_boards)]
        models = s.build_board_models_from_indices(idx_by_board, wd)
        excluded = {wd.word_to_index[w] for w in exclude if w in wd.word_to_index}
        scoring = np.array([i for i in wd.guess_indices if int(i) not in excluded],
                           dtype=np.int32)

        guess_no = len(history) + 1
        sizes = sorted(len(cand[b]) for b in active)
        print(f"\n=== Entscheidung vor Guess {guess_no} | aktive Boards={len(active)} | "
              f"Kandidaten min/median/max={sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}")

        t0 = time.perf_counter()
        score_fn = s.score_all_guesses_gpu if self.backend == "gpu" else s.score_all_guesses_cpu
        rows, pairs = score_fn(scoring, wd, models, active)
        elapsed = time.perf_counter() - t0

        rows = s.finalise_rows(rows, cand, active)
        chosen, reason = s.choose_next_guess(active, cand, rows)
        if chosen is None:
            raise BridgeError("Solver hat keinen Guess geliefert.")

        record, _ = s.build_decision_record(game_id, guess_no, chosen, reason,
                                            cand, solved, active)
        record["source"] = "live"
        s.append_decision_log(record)

        out = {
            "done": False,
            "guess": chosen["word"],
            "reason": reason,
            "guess_number": guess_no,
            "candidate": bool(chosen["is_candidate"]),
            "N": int(chosen["N"]),
            "SingletonCount": int(chosen.get("SingletonCount", 0)),
            "P_sum": float(chosen["P_sum"]),
            "ELog": float(chosen["ExpectedLogRemaining"]),
            "ERest": float(chosen["ExpectedRemaining"]),
            "InfoGain": float(chosen["InfoGain"]),
            "active_boards": len(active),
            "seconds": round(elapsed, 2),
            "pairs": int(pairs),
        }
        print(f">>> {chosen['word'].upper()} | {reason} | ELog={out['ELog']:.3f} | "
              f"N={out['N']} | {s.human_seconds(elapsed)}")

        if not self.args.no_cache:
            self.cache[key] = out
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache), encoding="utf-8")
            tmp.replace(self.cache_path)
        return out

    def finish(self, payload):
        """Spielergebnis protokollieren und unbekannte Zielwörter merken."""
        history = payload.get("history", [])
        _, solved, solved_at = self.reconstruct(history)
        got = [x for x in solved_at if x is not None]

        new_targets = sorted({history[k - 1]["guess"].lower() for k in got}
                             - self.base_solutions)
        if new_targets:
            with self.learned_path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(new_targets) + "\n")
            print(f"[Gelernt] {len(new_targets)} neue Zielwörter: {', '.join(new_targets)}")

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "game_id": payload.get("game_id"),
            "guesses_used": len(history),
            "solved": sum(solved),
            "boards": self.num_boards,
            "won": all(solved),
            "solved_at": solved_at,
            "avg_solve_guess": round(sum(got) / len(got), 4) if got else None,
            "words": [h["guess"] for h in history],
            "rejected_words": payload.get("exclude", []),
            "learned_targets": new_targets,
            "fingerprint": self.fingerprint,
        }
        with (OUTPUT_DIR / "live_games.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n### Spiel beendet: {record['solved']}/{self.num_boards} gelöst in "
              f"{record['guesses_used']} Guesses (Ø {record['avg_solve_guess']})")
        return record


BRIDGE = None


class Handler(BaseHTTPRequestHandler):
    """HTTP-Routen: GET /health, GET /duo-bridge.user.js, POST /next, POST /finish."""

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        """Antwort mit passendem Content-Type senden."""
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        """Request-Body lesen (Content-Length oder chunked)."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            data = b""
            while True:
                size = int(self.rfile.readline().strip().split(b";")[0], 16)
                if size == 0:
                    self.rfile.readline()
                    return data
                data += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        """Health-Check und Auslieferung des Userscripts."""
        if self.path == "/health":
            self._send(200, {"ok": True, "backend": BRIDGE.backend,
                             "fingerprint": BRIDGE.fingerprint})
        elif self.path in ("/duo-bridge.user.js", "/install"):
            self._send(200, (HERE / "duo-bridge.user.js").read_bytes(),
                       "text/javascript; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        """JSON-Anfragen an /next und /finish weiterreichen."""
        try:
            payload = json.loads(self._read_body() or b"{}")
            if self.path == "/next":
                self._send(200, BRIDGE.next_guess(payload))
            elif self.path == "/finish":
                self._send(200, BRIDGE.finish(payload))
            else:
                self._send(404, {"error": "not found"})
        except BridgeError as e:
            print(f"!!! {e}")
            self._send(422, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        """Zugriffslog nur ohne --quiet ausgeben."""
        if not BRIDGE.args.quiet:
            sys.stderr.write("[http] " + fmt % args + "\n")


def main():
    """Argumente lesen und Server auf 127.0.0.1 starten."""
    global BRIDGE
    ap = argparse.ArgumentParser(description="Duotrigordle Solver Bridge")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--solver", default=str(HERE / "duotrigordle_solver.py"))
    ap.add_argument("--solutions", default=str(DATA_DIR / "solutions_en.txt"))
    ap.add_argument("--guesses", default=str(DATA_DIR / "guesses_en.txt"))
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Guess-Wörter pro Batch (Standard aus dem Solver)")
    ap.add_argument("--wide-solutions", action="store_true",
                    help="Alle gültigen Wörter von Anfang an als mögliche Lösungen")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    BRIDGE = Bridge(args)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Lausche auf http://127.0.0.1:{args.port}  (Userscript: /duo-bridge.user.js)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    main()
