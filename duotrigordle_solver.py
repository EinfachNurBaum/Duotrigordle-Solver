#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duotrigordle Solver

Heuristischer Solver für Duotrigordle (32 Wordle-Boards gleichzeitig).
Ziel ist eine möglichst geringe Zahl an Versuchen, nicht minimale Laufzeit:
Pro Zug wird jedes erlaubte Wort gegen alle aktiven Boards bewertet.

Zustand:   S = (C_1, ..., C_32), C_b = noch mögliche Lösungen auf Board b
Kennzahlen pro Guess w:
    N          Boards, auf denen w Kandidat ist
    P_sum      Summe der Kandidatenwahrscheinlichkeiten
    P_any      Wahrscheinlichkeit, mindestens ein Board direkt zu lösen
    InfoGain   Entropie des Feedbacks über alle aktiven Boards
    ERest      erwartete Zahl verbleibender Kandidaten
    ELogRest   erwartete Restunsicherheit in log2-Kandidaten (Hauptkriterium)

Standalone:      python3 duotrigordle_solver.py   (simuliert Spiele)
Live im Browser: siehe duo_bridge_server.py und README.md
"""

import csv
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

# Numerische Libraries pro Prozess auf einen Thread begrenzen.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None


# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

NUM_BOARDS = 32
WORD_LENGTH = 5
MAX_GUESSES = 37

# Wortquellen: .txt (ein Wort pro Zeile) oder SQLite (.db/.sqlite).
DB_FILE = None
DB_TABLE = "word"
DB_COLUMN = "word"
TEXT_FILE = DATA_DIR / "guesses_en.txt"
SOLUTION_TEXT_FILE = DATA_DIR / "solutions_en.txt"
GUESS_TEXT_FILE = DATA_DIR / "guesses_en.txt"

# Optional: Datei mit exakt 32 Zielwörtern für ein reproduzierbares Testspiel.
TARGET_TEXT_FILE = None

RANDOM_SEED = 42

PRINT_INITIAL_PROBABILITIES = False
PRINT_TOP_N = 10
REVEAL_TARGETS = True
PRINT_PROGRESS_EVERY_BATCH = True

# "auto" (GPU falls CuPy vorhanden), "cpu" oder "gpu".
COMPUTATION_BACKEND = "auto"

CPU_FRACTION = 0.75
MIN_CPU_WORKERS = 1
MAX_CPU_WORKERS = None
GUESS_BATCH_SIZE = 1000
WORKER_NICENESS = 5

# Gewichte der Prior-Wahrscheinlichkeit p_b(w).
WEIGHTS = {
    "word_frequency": 0.20,
    "letter_frequency": 0.30,
    "position_frequency": 0.30,
    "corpus_frequency": 0.20,
}

USE_WORDFREQ = True
WORDFREQ_LANGUAGE = "en"

USE_ATTEMPT_ORIENTED_SELECTION = True
ATTEMPT_REST_MARGIN = 0.00
PREFER_SINGLETON_BOARDS = True
SINGLETON_HARD_PRIORITY = True

DECISION_LOG_FILE = OUTPUT_DIR / "decisions.jsonl"
SINGLETON_LOG_FILE = OUTPUT_DIR / "singletons.csv"
LOG_SINGLETON_BOARD_MAP = True

# 1 = ein Spiel, >1 = Benchmark über mehrere zufällige Spiele.
BENCHMARK_GAMES = 1

random.seed(RANDOM_SEED)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def normalize_word(word: str) -> str:
    """Wort trimmen und kleinschreiben."""
    return str(word).strip().lower()


def is_valid_word(word: str) -> bool:
    """True, wenn das Wort genau WORD_LENGTH Buchstaben hat."""
    return len(word) == WORD_LENGTH and word.isalpha()


def chunked(seq, size):
    """Sequenz in Blöcke fester Größe zerlegen."""
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def human_seconds(seconds: float) -> str:
    """Sekunden lesbar formatieren (z. B. '3m 12s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


# ============================================================
# WORTLISTEN
# ============================================================

def load_single_word_source(path: Path):
    """Wörter aus .txt oder SQLite laden; liefert (Liste, Häufigkeiten)."""
    path = Path(path)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(f"SELECT {DB_COLUMN} FROM {DB_TABLE}").fetchall()
        raw = [normalize_word(row[0]) for row in rows if row and row[0] is not None]
    else:
        raw = [normalize_word(line)
               for line in path.read_text(encoding="utf-8").splitlines()]

    raw = [w for w in raw if is_valid_word(w)]
    return raw, Counter(raw)


def load_word_data():
    """Lösungs- und Guess-Liste gemäß Konfiguration laden."""
    source_path = Path(DB_FILE) if DB_FILE and Path(DB_FILE).exists() else Path(TEXT_FILE)
    if not source_path.exists():
        raise FileNotFoundError(f"Keine Wortliste gefunden: {source_path.name}")

    print(f"Lese Wortliste aus {source_path.name} ...")
    raw_words, source_counts = load_single_word_source(source_path)

    solution_counts = source_counts.copy()
    guess_counts = source_counts.copy()
    solution_words = sorted(set(raw_words))
    guess_words = sorted(set(raw_words))

    if SOLUTION_TEXT_FILE:
        p = Path(SOLUTION_TEXT_FILE)
        if not p.exists():
            raise FileNotFoundError(f"Solution-Datei nicht gefunden: {p.name}")
        raw_solution, solution_counts = load_single_word_source(p)
        solution_words = sorted(set(raw_solution))

    if GUESS_TEXT_FILE:
        p = Path(GUESS_TEXT_FILE)
        if not p.exists():
            raise FileNotFoundError(f"Guess-Datei nicht gefunden: {p.name}")
        raw_guess, guess_counts = load_single_word_source(p)
        guess_words = sorted(set(raw_guess))

    if len(solution_words) < NUM_BOARDS:
        raise ValueError(f"Zu wenige Lösungswörter: {len(solution_words)}.")
    if not guess_words:
        raise ValueError("Keine Guess-Wörter gefunden.")

    print(f"{len(solution_words)} Lösungswörter, {len(guess_words)} Guess-Wörter")
    return solution_words, solution_counts, guess_words, guess_counts


def load_targets(solution_words):
    """32 Zielwörter aus TARGET_TEXT_FILE oder zufällig wählen."""
    if TARGET_TEXT_FILE:
        p = Path(TARGET_TEXT_FILE)
        if not p.exists():
            raise FileNotFoundError(f"Target-Datei nicht gefunden: {p.name}")
        raw = [normalize_word(x) for x in p.read_text(encoding="utf-8").splitlines()]
        targets = [w for w in raw if is_valid_word(w)]
        if len(targets) != NUM_BOARDS:
            raise ValueError(f"Target-Datei braucht {NUM_BOARDS} Wörter, "
                             f"gefunden: {len(targets)}")
        missing = [w for w in targets if w not in solution_words]
        if missing:
            raise ValueError("Targets nicht in der Lösungsliste: " + ", ".join(missing))
        return targets

    return random.sample(solution_words, NUM_BOARDS)


# ============================================================
# WORDLE-FEEDBACK  (0 = grau, 1 = gelb, 2 = grün)
# ============================================================

GREEN_CODE = 242  # 22222 zur Basis 3


def evaluate_guess(guess: str, target: str):
    """Exaktes Wordle-Feedback inklusive doppelter Buchstaben."""
    result = [0] * WORD_LENGTH
    remaining = Counter(target)

    for i in range(WORD_LENGTH):
        if guess[i] == target[i]:
            result[i] = 2
            remaining[guess[i]] -= 1

    for i in range(WORD_LENGTH):
        if result[i] != 0:
            continue
        char = guess[i]
        if remaining[char] > 0:
            result[i] = 1
            remaining[char] -= 1

    return tuple(result)


def feedback_to_code(feedback):
    """Feedback-Tupel als Zahl 0..242 kodieren."""
    code = 0
    factor = 1
    for x in feedback:
        code += int(x) * factor
        factor *= 3
    return code


def filter_candidates(candidates, guess, feedback):
    """Kandidaten behalten, die mit dem beobachteten Feedback vereinbar sind."""
    return {word for word in candidates if evaluate_guess(guess, word) == feedback}


# ============================================================
# NUMERISCHE WORTDARSTELLUNG
# ============================================================

class WordData:
    """Alle Wörter als Integer-Matrix plus globale Häufigkeitsmerkmale."""

    def __init__(self, solution_words, solution_counts, guess_words, guess_counts):
        if np is None:
            raise RuntimeError("NumPy wird benötigt: pip install numpy")

        self.solution_words = solution_words
        self.solution_counts = solution_counts
        self.guess_words = guess_words
        self.guess_counts = guess_counts

        self.all_words = sorted(set(guess_words) | set(solution_words))
        self.word_to_index = {w: i for i, w in enumerate(self.all_words)}
        self.index_to_word = self.all_words

        self.alphabet = sorted(set("".join(self.all_words)))
        self.char_to_int = {c: i for i, c in enumerate(self.alphabet)}

        self.codes = np.array(
            [[self.char_to_int[c] for c in word] for word in self.all_words],
            dtype=np.uint8,
        )
        self.n_words = len(self.all_words)

        self.solution_indices = np.array(
            [self.word_to_index[w] for w in self.solution_words], dtype=np.int32)
        self.guess_indices = np.array(
            [self.word_to_index[w] for w in self.guess_words], dtype=np.int32)

        self.word_frequency = self._global_source_frequency()
        self.corpus_frequency = self._global_corpus_frequency()
        self.unique_letter_count = np.array(
            [len(set(w)) for w in self.all_words], dtype=np.uint8)
        self.repeated_letter_count = WORD_LENGTH - self.unique_letter_count

    def _global_source_frequency(self):
        """Log-normierte Häufigkeit eines Wortes in der Quellliste."""
        values = np.array([self.guess_counts.get(w, 0) for w in self.all_words],
                          dtype=np.float64)
        positive = values[values > 0]
        if len(positive) == 0 or positive.max() <= 0:
            return np.zeros(self.n_words, dtype=np.float64)
        return np.log1p(values) / math.log1p(float(positive.max()))

    def _global_corpus_frequency(self):
        """Normierte Zipf-Häufigkeit aus wordfreq (0, falls nicht installiert)."""
        if not USE_WORDFREQ or zipf_frequency is None:
            return np.zeros(self.n_words, dtype=np.float64)
        values = np.array(
            [max(0.0, float(zipf_frequency(w, WORDFREQ_LANGUAGE))) for w in self.all_words],
            dtype=np.float64,
        )
        max_value = float(values.max()) if len(values) else 0.0
        if max_value <= 0:
            return np.zeros(self.n_words, dtype=np.float64)
        return values / max_value


# ============================================================
# BOARD-MODELL
# ============================================================

def compute_board_model(candidate_indices, word_data: WordData):
    """Prior p_b(w) sowie Buchstaben-/Positionsstatistik für ein Board berechnen."""
    n_total = len(candidate_indices)
    n_words = word_data.n_words
    probs_dense = np.zeros(n_words, dtype=np.float64)

    if n_total == 0:
        return {
            "candidate_indices": np.array([], dtype=np.int32),
            "candidate_mask": np.zeros(n_words, dtype=np.bool_),
            "probabilities": probs_dense,
            "letter_frequency": np.zeros(len(word_data.alphabet), dtype=np.float64),
            "position_frequency": np.zeros((WORD_LENGTH, len(word_data.alphabet)),
                                           dtype=np.float64),
            "is_singleton": False,
            "singleton_index": -1,
        }

    codes = word_data.codes[candidate_indices]
    alphabet_size = len(word_data.alphabet)

    letter_presence = np.zeros(alphabet_size, dtype=np.float64)
    for row in codes:
        for char in np.unique(row):
            letter_presence[int(char)] += 1.0
    letter_frequency = letter_presence / n_total

    position_frequency = np.zeros((WORD_LENGTH, alphabet_size), dtype=np.float64)
    for pos in range(WORD_LENGTH):
        chars, counts = np.unique(codes[:, pos], return_counts=True)
        position_frequency[pos, chars] = counts / n_total

    word_freq = word_data.word_frequency[candidate_indices]
    corpus_freq = word_data.corpus_frequency[candidate_indices]

    letter_mean = np.zeros(n_total, dtype=np.float64)
    position_mean = np.zeros(n_total, dtype=np.float64)
    for i, row in enumerate(codes):
        letter_mean[i] = float(letter_frequency[np.unique(row)].mean())
        position_mean[i] = float(np.mean(
            [position_frequency[pos, int(ch)] for pos, ch in enumerate(row)]))

    active_weights = dict(WEIGHTS)
    if not USE_WORDFREQ or zipf_frequency is None:
        active_weights["corpus_frequency"] = 0.0
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0:
        raise ValueError("Alle Wahrscheinlichkeitsgewichte sind 0.")
    active_weights = {k: v / weight_sum for k, v in active_weights.items()}

    raw = (
        active_weights["word_frequency"] * word_freq
        + active_weights["letter_frequency"] * letter_mean
        + active_weights["position_frequency"] * position_mean
        + active_weights["corpus_frequency"] * corpus_freq
    )
    raw = np.maximum(raw, 0.0)
    total = float(raw.sum())
    if total <= 0:
        raw[:] = 1.0
        total = float(raw.sum())

    probs_dense[candidate_indices] = raw / total

    candidate_mask = np.zeros(n_words, dtype=np.bool_)
    candidate_mask[candidate_indices] = True

    is_singleton = n_total == 1
    return {
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "probabilities": probs_dense,
        "letter_frequency": letter_frequency,
        "position_frequency": position_frequency,
        "is_singleton": is_singleton,
        "singleton_index": int(candidate_indices[0]) if is_singleton else -1,
    }


# ============================================================
# FEEDBACK IM BATCH (NumPy / CuPy)
# ============================================================

def numpy_feedback_codes(guess_codes, target_codes):
    """Feedback-Codes (B x N) für B Guesses gegen N Targets, exakt bei Doppelbuchstaben."""
    B = guess_codes.shape[0]
    N = target_codes.shape[0]

    green = guess_codes[:, None, :] == target_codes[None, :, :]
    codes = np.zeros((B, N), dtype=np.int16)
    factors = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    codes += (green.astype(np.int16) * 2) @ factors

    yellow = np.zeros((B, N, WORD_LENGTH), dtype=np.bool_)
    for i in range(WORD_LENGTH):
        guess_char = guess_codes[:, None, i]
        same = target_codes[None, :, :] == guess_char[:, :, None]
        available = np.sum(same, axis=2) - np.sum(green & same, axis=2)

        already_used = np.zeros((B, N), dtype=np.int16)
        for j in range(i):
            same_char = guess_codes[:, j][:, None] == guess_codes[:, i][:, None]
            already_used += (yellow[:, :, j] & same_char).astype(np.int16)

        yellow[:, :, i] = (~green[:, :, i]) & (already_used < available)

    codes += (yellow.astype(np.int16) * factors).sum(axis=2)
    return codes


def cupy_feedback_codes(guess_codes, target_codes):
    """GPU-Variante von numpy_feedback_codes."""
    B = guess_codes.shape[0]
    N = target_codes.shape[0]

    green = guess_codes[:, None, :] == target_codes[None, :, :]
    factors = cp.asarray([1, 3, 9, 27, 81], dtype=cp.int16)
    codes = (green.astype(cp.int16) * 2) @ factors

    yellow = cp.zeros((B, N, WORD_LENGTH), dtype=cp.bool_)
    for i in range(WORD_LENGTH):
        guess_char = guess_codes[:, None, i]
        same = target_codes[None, :, :] == guess_char[:, :, None]
        available = cp.sum(same, axis=2) - cp.sum(green & same, axis=2)

        already_used = cp.zeros((B, N), dtype=cp.int16)
        for j in range(i):
            same_char = guess_codes[:, j][:, None] == guess_codes[:, i][:, None]
            already_used += (yellow[:, :, j] & same_char).astype(cp.int16)

        yellow[:, :, i] = (~green[:, :, i]) & (already_used < available)

    codes += (yellow.astype(cp.int16) * factors).sum(axis=2)
    return codes


def pattern_metrics(counts, pattern_probs):
    """InfoGain, ERest und ELogRest aus Musterhäufigkeiten (ohne Komplett-Grün)."""
    positive = pattern_probs > 0
    info_gain = float(-np.sum(pattern_probs[positive] * np.log2(pattern_probs[positive])))

    counts = counts.astype(np.float64).copy()
    probs = pattern_probs.copy()
    counts[GREEN_CODE] = 0.0
    probs[GREEN_CODE] = 0.0

    expected_remaining = float(np.sum(probs * counts))
    nz = counts > 0
    expected_log_remaining = (
        float(np.sum(probs[nz] * np.log2(counts[nz]))) if np.any(nz) else 0.0
    )
    return info_gain, expected_remaining, expected_log_remaining


def score_guess_on_board(guess_index, board_model, word_data):
    """Feedbackverteilung eines Guess auf einem Board; Singletons tragen nichts bei."""
    candidate_indices = board_model["candidate_indices"]
    if len(candidate_indices) <= 1:
        return {"info_gain": 0.0, "expected_remaining": 0.0, "expected_log_remaining": 0.0}

    pattern_codes = numpy_feedback_codes(
        word_data.codes[guess_index:guess_index + 1],
        word_data.codes[candidate_indices],
    )[0]

    counts = np.bincount(pattern_codes, minlength=243)
    probs = np.bincount(pattern_codes,
                        weights=board_model["probabilities"][candidate_indices],
                        minlength=243)
    ig, er, elr = pattern_metrics(counts, probs)
    return {"info_gain": ig, "expected_remaining": er, "expected_log_remaining": elr}


# ============================================================
# SCORING (CPU-Worker / GPU)
# ============================================================

_WORKER_STATE = None


def _worker_init():
    """Worker-Prozess unter Linux etwas niedriger priorisieren."""
    try:
        if sys.platform.startswith("linux") and WORKER_NICENESS:
            os.nice(WORKER_NICENESS)
    except Exception:
        pass


def _empty_row(word_data, guess_index):
    """Leere Ergebniszeile für einen Guess anlegen."""
    return {
        "word": word_data.index_to_word[guess_index],
        "index": guess_index,
        "N": 0,
        "SingletonCount": 0,
        "P_sum": 0.0,
        "P_any": 0.0,
        "L": 0.0,
        "Position": 0.0,
        "WordFreq": float(word_data.word_frequency[guess_index]),
        "Corpus": float(word_data.corpus_frequency[guess_index]),
        "InfoGain": 0.0,
        "ExpectedRemaining": 0.0,
        "ExpectedLogRemaining": 0.0,
        "UniqueLetters": int(word_data.unique_letter_count[guess_index]),
        "RepeatedLetters": int(word_data.repeated_letter_count[guess_index]),
        "_P_not": 1.0,
    }


def _add_candidate_features(row, guess_index, model, word_data):
    """Candidate-, Singleton-, Buchstaben- und Positionsmerkmale eines Boards addieren."""
    if model["candidate_mask"][guess_index]:
        p = float(model["probabilities"][guess_index])
        row["N"] += 1
        row["P_sum"] += p
        row["_P_not"] *= (1.0 - p)
        if model["is_singleton"] and model["singleton_index"] == guess_index:
            row["SingletonCount"] += 1

    guess_codes = word_data.codes[guess_index]
    row["L"] += float(model["letter_frequency"][np.unique(guess_codes)].sum())
    row["Position"] += float(np.sum(
        [model["position_frequency"][pos, int(ch)] for pos, ch in enumerate(guess_codes)]))


def _score_guess_batch_worker(batch):
    """CPU-Worker: einen Block Guesses gegen alle aktiven Boards bewerten."""
    word_data, board_models, active_boards, _ = _WORKER_STATE
    rows = []
    pair_evaluations = 0

    for guess_index in batch:
        guess_index = int(guess_index)
        row = _empty_row(word_data, guess_index)

        for b in active_boards:
            model = board_models[b]
            n = len(model["candidate_indices"])
            if n == 0:
                continue
            pair_evaluations += n
            _add_candidate_features(row, guess_index, model, word_data)

            score = score_guess_on_board(guess_index, model, word_data)
            row["InfoGain"] += score["info_gain"]
            row["ExpectedRemaining"] += score["expected_remaining"]
            row["ExpectedLogRemaining"] += score["expected_log_remaining"]

        row["P_any"] = 1.0 - row.pop("_P_not")
        rows.append(row)

    return rows, pair_evaluations


def gpu_score_batch(guess_indices, board_models, active_boards, word_data):
    """GPU: einen Block Guesses gegen alle aktiven Boards bewerten."""
    if cp is None:
        raise RuntimeError("CuPy ist nicht installiert.")

    guess_indices_np = np.asarray(guess_indices, dtype=np.int32)
    rows = [_empty_row(word_data, int(g)) for g in guess_indices_np]
    total_pairs = 0
    guess_codes_gpu = cp.asarray(word_data.codes[guess_indices_np])

    for b in active_boards:
        model = board_models[b]
        candidate_indices = model["candidate_indices"]
        n = len(candidate_indices)
        if n == 0:
            continue

        for local_i, guess_index in enumerate(guess_indices_np):
            _add_candidate_features(rows[local_i], int(guess_index), model, word_data)

        if n == 1:
            continue

        targets_gpu = cp.asarray(word_data.codes[candidate_indices])
        probs_gpu = cp.asarray(model["probabilities"][candidate_indices])
        pattern_codes = cupy_feedback_codes(guess_codes_gpu, targets_gpu)
        total_pairs += len(guess_indices_np) * n

        for local_i in range(len(guess_indices_np)):
            patterns = pattern_codes[local_i]
            counts = cp.asnumpy(cp.bincount(patterns, minlength=243))
            probs = cp.asnumpy(cp.bincount(patterns, weights=probs_gpu, minlength=243))
            ig, er, elr = pattern_metrics(counts, probs)
            row = rows[local_i]
            row["InfoGain"] += ig
            row["ExpectedRemaining"] += er
            row["ExpectedLogRemaining"] += elr

    for row in rows:
        row["P_any"] = 1.0 - row.pop("_P_not")
    return rows, total_pairs


def choose_backend():
    """Rechenbackend gemäß COMPUTATION_BACKEND wählen."""
    requested = COMPUTATION_BACKEND.lower()
    if requested == "gpu":
        if cp is None:
            raise RuntimeError("COMPUTATION_BACKEND='gpu', aber CuPy fehlt.")
        return "gpu"
    if requested == "cpu":
        if np is None:
            raise RuntimeError("NumPy fehlt: pip install numpy")
        return "cpu"
    if requested == "auto":
        if cp is not None:
            return "gpu"
        if np is not None:
            return "cpu"
    raise RuntimeError("Weder NumPy noch CuPy verfügbar.")


def cpu_workers():
    """Anzahl CPU-Worker (CPU_FRACTION der logischen Kerne)."""
    workers = max(MIN_CPU_WORKERS, int((os.cpu_count() or 1) * CPU_FRACTION))
    if MAX_CPU_WORKERS is not None:
        workers = min(workers, MAX_CPU_WORKERS)
    return max(1, workers)


def _print_progress(label, batch_no, n_batches, n_words, total_pairs, start):
    """Fortschritt eines Batches ausgeben."""
    if PRINT_PROGRESS_EVERY_BATCH:
        print(f"    {label}-Batch {batch_no:4d}/{n_batches:4d} | "
              f"Wörter={min(batch_no * GUESS_BATCH_SIZE, n_words):5d}/{n_words:5d} | "
              f"Pairs={total_pairs:,} | Zeit={human_seconds(time.perf_counter() - start)}",
              flush=True)


def score_all_guesses_cpu(guess_indices, word_data, board_models, active_boards):
    """Alle Guesses auf der CPU bewerten (Linux: fork-Pool, sonst Threads)."""
    global _WORKER_STATE
    _WORKER_STATE = (word_data, board_models, active_boards, None)
    batches = list(chunked(guess_indices, GUESS_BATCH_SIZE))
    start = time.perf_counter()
    total_pairs = 0
    all_rows = []

    if sys.platform.startswith("linux"):
        with get_context("fork").Pool(processes=cpu_workers(),
                                      initializer=_worker_init) as pool:
            results = pool.imap(_score_guess_batch_worker, batches, chunksize=1)
            for batch_no, (rows, pairs) in enumerate(results, start=1):
                all_rows.extend(rows)
                total_pairs += pairs
                _print_progress("CPU", batch_no, len(batches), len(guess_indices),
                                total_pairs, start)
    else:
        with ThreadPoolExecutor(max_workers=cpu_workers()) as executor:
            futures = [executor.submit(_score_guess_batch_worker, b) for b in batches]
            for batch_no, future in enumerate(futures, start=1):
                rows, pairs = future.result()
                all_rows.extend(rows)
                total_pairs += pairs
                _print_progress("Thread", batch_no, len(batches), len(guess_indices),
                                total_pairs, start)

    return all_rows, total_pairs


def score_all_guesses_gpu(guess_indices, word_data, board_models, active_boards):
    """Alle Guesses blockweise auf der GPU bewerten."""
    if cp is None:
        raise RuntimeError("CuPy nicht verfügbar.")
    batches = list(chunked(guess_indices, GUESS_BATCH_SIZE))
    start = time.perf_counter()
    total_pairs = 0
    all_rows = []

    for batch_no, batch in enumerate(batches, start=1):
        rows, pairs = gpu_score_batch(batch, board_models, active_boards, word_data)
        all_rows.extend(rows)
        total_pairs += pairs
        cp.cuda.Stream.null.synchronize()
        _print_progress("GPU", batch_no, len(batches), len(guess_indices),
                        total_pairs, start)

    return all_rows, total_pairs


# ============================================================
# RANKING UND AUSWAHL
# ============================================================

def finalise_rows(rows, candidates_by_board, active_boards):
    """Candidate/Probe markieren und abgeleitete Kennzahlen ergänzen."""
    current_total = sum(len(candidates_by_board[b]) for b in active_boards)
    for row in rows:
        row["SingletonCount"] = int(row.get("SingletonCount", 0))
        row["is_candidate"] = row["N"] > 0
        row["is_probe"] = row["N"] == 0
        row["ExpectedRemainingRatio"] = (
            row["ExpectedRemaining"] / current_total if current_total > 0 else 0.0)
        row["SolveBoardsExpected"] = row["P_sum"]
    return rows


def rank_candidate_guesses(rows):
    """Candidate-Guesses nach Singletons, N, P_sum, ... absteigend sortieren."""
    candidates = [r for r in rows if r["is_candidate"]]
    candidates.sort(key=lambda r: (
        r["SingletonCount"], r["N"], r["P_sum"], r["P_any"], r["InfoGain"],
        r["L"], r["Position"], r["Corpus"], r["WordFreq"],
    ), reverse=True)
    return candidates


def rank_probe_guesses(rows):
    """Probe-Guesses nach geringster Restunsicherheit sortieren."""
    probes = [r for r in rows if r["is_probe"]]
    probes.sort(key=lambda r: (
        r["ExpectedLogRemaining"], -r["InfoGain"], -r["L"], -r["Position"], -r["Corpus"],
    ))
    return probes


def rank_attempts(rows):
    """Hauptranking: ELogRest minimieren, danach P_sum, N, P_any, InfoGain."""
    return sorted(rows, key=lambda r: (
        r["ExpectedLogRemaining"], -r["P_sum"], -r["N"], -r["P_any"],
        -r["InfoGain"], -r["L"], -r["Position"], -r["Corpus"],
    ))


def choose_next_guess(active_boards, candidate_sets, all_scores):
    """Nächsten Guess wählen; bekannte Singleton-Boards haben Vorrang."""
    if not active_boards:
        return None, "finished"

    has_singleton = any(len(candidate_sets[b]) == 1 for b in active_boards)
    if PREFER_SINGLETON_BOARDS and SINGLETON_HARD_PRIORITY and has_singleton:
        singleton_scores = [r for r in all_scores if r.get("SingletonCount", 0) > 0]
        if singleton_scores:
            best = min(singleton_scores, key=lambda r: (
                -r["SingletonCount"], r["ExpectedLogRemaining"], -r["P_sum"],
                -r["N"], -r["InfoGain"], -r["L"], -r["Position"],
            ))
            return best, "candidate-singleton-priority"

    if not all_scores:
        return None, "none"

    if USE_ATTEMPT_ORIENTED_SELECTION:
        best = rank_attempts(all_scores)[0]
        if best["is_probe"] and ATTEMPT_REST_MARGIN > 0:
            candidate_rows = [r for r in all_scores if r["is_candidate"]]
            if candidate_rows:
                best_candidate = min(candidate_rows, key=lambda r: (
                    r["ExpectedLogRemaining"], -r["P_sum"], -r["N"],
                    -r.get("SingletonCount", 0),
                ))
                if (best_candidate["ExpectedLogRemaining"]
                        <= best["ExpectedLogRemaining"] + ATTEMPT_REST_MARGIN):
                    return best_candidate, "candidate-within-rest-margin"
        return best, "attempt-oriented"

    candidate_rows = rank_candidate_guesses(all_scores)
    if candidate_rows:
        return candidate_rows[0], "candidate-hierarchy"
    probe_rows = rank_probe_guesses(all_scores)
    return (probe_rows[0], "probe-only") if probe_rows else (None, "none")


# ============================================================
# AUSGABE
# ============================================================

def format_feedback(feedback):
    """Feedback als 'G', 'Y', '.' darstellen."""
    return "".join("G" if x == 2 else "Y" if x == 1 else "." for x in feedback)


def print_initial_probabilities(candidates_by_board, board_models, word_data):
    """Top-Kandidaten jedes Boards im Anfangszustand ausgeben."""
    print("\n" + "=" * 100 + "\nANFANGSZUSTAND – PRIOR-VERTEILUNGEN\n" + "=" * 100)
    for b in range(NUM_BOARDS):
        model = board_models[b]
        candidate_indices = model["candidate_indices"]
        probs = model["probabilities"][candidate_indices]
        print(f"\nBoard {b + 1:02d}: {len(candidates_by_board[b])} Kandidaten")
        for rank, local_i in enumerate(np.argsort(-probs)[:PRINT_TOP_N], start=1):
            idx = int(candidate_indices[local_i])
            print(f"  {rank:2d}. {word_data.index_to_word[idx]:<8} "
                  f"p={float(probs[local_i]):.6f} | "
                  f"Korpus={word_data.corpus_frequency[idx]:.3f}")


def print_guess_ranking(candidate_scores, probe_scores, attempt_scores):
    """Top-Candidate-, Probe- und Gesamt-Ranking ausgeben."""
    print("\n  TOP CANDIDATE GUESSES")
    for i, r in enumerate(candidate_scores[:PRINT_TOP_N], start=1):
        print(f"    {i:2d}. {r['word']:<8} S={r['SingletonCount']:2d} N={r['N']:2d} "
              f"Psum={r['P_sum']:.4f} IG={r['InfoGain']:.3f} "
              f"ELog={r['ExpectedLogRemaining']:.3f}")

    print("\n  TOP PROBE GUESSES")
    for i, r in enumerate(probe_scores[:PRINT_TOP_N], start=1):
        print(f"    {i:2d}. {r['word']:<8} IG={r['InfoGain']:.3f} "
              f"ERest={r['ExpectedRemaining']:.2f} ELog={r['ExpectedLogRemaining']:.3f}")

    print("\n  TOP GUESSES – ATTEMPT-ORIENTIERT")
    for i, r in enumerate(attempt_scores[:PRINT_TOP_N], start=1):
        cls = "CAND" if r["is_candidate"] else "PROBE"
        print(f"    {i:2d}. {r['word']:<8} [{cls}] ELog={r['ExpectedLogRemaining']:.3f} "
              f"Psum={r['P_sum']:.4f} N={r['N']:2d} IG={r['InfoGain']:.3f}")


def print_board_results(guess_number, guess, targets, feedbacks, candidates_by_board, solved):
    """Feedback und Kandidatenzahl aller Boards nach einem Guess ausgeben."""
    print("\n" + "-" * 100 + f"\nGUESS {guess_number}: {guess.upper()}\n" + "-" * 100)
    for b in range(NUM_BOARDS):
        pattern = "-----" if feedbacks[b] is None else format_feedback(feedbacks[b])
        target_text = targets[b].upper() if REVEAL_TARGETS else "?????"
        status = "SOLVED" if solved[b] else "active"
        print(f"Board {b + 1:02d} | Target={target_text:<6} | Feedback={pattern} | "
              f"Candidates={len(candidates_by_board[b]):4d} | {status}")


# ============================================================
# ZUSTAND UND LOGGING
# ============================================================

def build_initial_candidate_state(word_data):
    """Jedes Board startet mit der kompletten Lösungsliste."""
    return [np.array(word_data.solution_indices, dtype=np.int32).copy()
            for _ in range(NUM_BOARDS)]


def build_board_models_from_indices(candidate_indices_by_board, word_data):
    """Board-Modelle für alle Boards berechnen."""
    return [compute_board_model(indices, word_data) for indices in candidate_indices_by_board]


def convert_candidate_indices_to_sets(candidate_indices_by_board, word_data):
    """Index-Arrays in Wortmengen umwandeln."""
    return [{word_data.index_to_word[int(i)] for i in indices}
            for indices in candidate_indices_by_board]


def build_singleton_map(candidate_sets, solved):
    """{Wort: [Boardnummern]} aller ungelösten Boards mit genau einem Kandidaten."""
    mapping = {}
    for b in range(NUM_BOARDS):
        if not solved[b] and len(candidate_sets[b]) == 1:
            mapping.setdefault(next(iter(candidate_sets[b])), []).append(b + 1)
    return mapping


def append_decision_log(record):
    """Eine Entscheidung als JSON-Zeile anhängen."""
    path = Path(DECISION_LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_singleton_log(record):
    """Kompakten Singleton-Eintrag als CSV-Zeile anhängen."""
    path = Path(SINGLETON_LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "game_id", "guess_number", "guess", "singleton_count",
              "singleton_unique_words", "singleton_words", "selected_reason",
              "active_boards", "solved_boards_before"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fields})


def build_decision_record(game_id, guess_number, chosen, reason, candidate_sets,
                          solved, active_boards):
    """Logeintrag für einen gewählten Guess zusammenstellen."""
    smap = build_singleton_map(candidate_sets, solved)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "game_id": game_id,
        "guess_number": int(guess_number),
        "guess": chosen["word"],
        "selected_reason": reason,
        "candidate": bool(chosen["is_candidate"]),
        "N": int(chosen["N"]),
        "P_sum": float(chosen["P_sum"]),
        "P_any": float(chosen["P_any"]),
        "InfoGain": float(chosen["InfoGain"]),
        "ExpectedRemaining": float(chosen["ExpectedRemaining"]),
        "ExpectedLogRemaining": float(chosen["ExpectedLogRemaining"]),
        "LetterScore": float(chosen["L"]),
        "PositionScore": float(chosen["Position"]),
        "WordFreq": float(chosen["WordFreq"]),
        "Corpus": float(chosen["Corpus"]),
        "UniqueLetters": int(chosen["UniqueLetters"]),
        "SingletonCount": int(chosen.get("SingletonCount", 0)),
        "SingletonBoardsForGuess": smap.get(chosen["word"], []),
        "SingletonUniqueWordCountBeforeGuess": len(smap),
        "SingletonBoardCountBeforeGuess": sum(len(v) for v in smap.values()),
        "ActiveBoardCountBeforeGuess": len(active_boards),
        "SolvedBoardCountBeforeGuess": int(sum(solved)),
    }
    if LOG_SINGLETON_BOARD_MAP:
        record["SingletonBoardMap"] = smap
    return record, smap


# ============================================================
# SIMULATION
# ============================================================

def play_game(word_data, targets, backend, game_id=1):
    """Ein komplettes Duotrigordle gegen bekannte Zielwörter simulieren."""
    print("\n" + "#" * 100 + "\nNEUES DUOTRIGORDLE\n" + "#" * 100)
    print(f"Backend: {backend} | Worker: {cpu_workers() if backend == 'cpu' else 'GPU'} "
          f"| Batchgröße: {GUESS_BATCH_SIZE}")

    candidate_indices_by_board = build_initial_candidate_state(word_data)
    candidate_sets = convert_candidate_indices_to_sets(candidate_indices_by_board, word_data)
    solved = [False] * NUM_BOARDS
    solution_guess_number = [None] * NUM_BOARDS

    if PRINT_INITIAL_PROBABILITIES:
        board_models = build_board_models_from_indices(candidate_indices_by_board, word_data)
        print_initial_probabilities(candidate_sets, board_models, word_data)

    if REVEAL_TARGETS:
        print("\nTargets: " + ", ".join(t.upper() for t in targets))

    total_pair_evaluations = 0

    for guess_number in range(1, MAX_GUESSES + 1):
        active_boards = [b for b in range(NUM_BOARDS) if not solved[b]]
        if not active_boards:
            break

        board_models = build_board_models_from_indices(candidate_indices_by_board, word_data)
        print("\n" + "=" * 100 + f"\nENTSCHEIDUNG VOR GUESS {guess_number}\n" + "=" * 100)

        score_start = time.perf_counter()
        score_fn = score_all_guesses_gpu if backend == "gpu" else score_all_guesses_cpu
        rows, pairs = score_fn(word_data.guess_indices, word_data, board_models, active_boards)
        total_pair_evaluations += pairs

        rows = finalise_rows(rows, candidate_sets, active_boards)
        print(f"\nScoring: {pairs:,} Paare in "
              f"{human_seconds(time.perf_counter() - score_start)}")
        print_guess_ranking(rank_candidate_guesses(rows), rank_probe_guesses(rows),
                            rank_attempts(rows))

        chosen, reason = choose_next_guess(active_boards, candidate_sets, rows)
        if chosen is None:
            print("Kein Guess mehr möglich.")
            break

        record, smap = build_decision_record(game_id, guess_number, chosen, reason,
                                             candidate_sets, solved, active_boards)
        append_decision_log(record)
        append_singleton_log({
            "timestamp": record["timestamp"],
            "game_id": game_id,
            "guess_number": guess_number,
            "guess": chosen["word"],
            "singleton_count": chosen.get("SingletonCount", 0),
            "singleton_unique_words": len(smap),
            "singleton_words": ";".join(f"{w}:{','.join(map(str, bs))}"
                                        for w, bs in sorted(smap.items())),
            "selected_reason": reason,
            "active_boards": len(active_boards),
            "solved_boards_before": sum(solved),
        })

        guess = chosen["word"]
        print(f"\n>>> GEWÄHLT: {guess.upper()} | {reason} | "
              f"ELog={chosen['ExpectedLogRemaining']:.4f} | N={chosen['N']}")

        feedbacks = [None] * NUM_BOARDS
        guess_index = word_data.word_to_index[guess]
        for b in active_boards:
            feedback = evaluate_guess(guess, targets[b])
            feedbacks[b] = feedback
            if feedback == (2, 2, 2, 2, 2):
                solved[b] = True
                solution_guess_number[b] = guess_number
                candidate_indices_by_board[b] = np.array([guess_index], dtype=np.int32)
                candidate_sets[b] = {guess}
            else:
                new_set = filter_candidates(candidate_sets[b], guess, feedback)
                candidate_sets[b] = new_set
                candidate_indices_by_board[b] = np.array(
                    [word_data.word_to_index[w] for w in sorted(new_set)], dtype=np.int32)

        print_board_results(guess_number, guess, targets, feedbacks, candidate_sets, solved)

        empty = [b + 1 for b in range(NUM_BOARDS) if not solved[b] and not candidate_sets[b]]
        if empty:
            raise RuntimeError("Kandidatenmenge leer auf Boards: " + ", ".join(map(str, empty)))

    solved_count = sum(solved)
    print("\n" + "=" * 100 + "\nENDERGEBNIS\n" + "=" * 100)
    print(f"{solved_count}/{NUM_BOARDS} Boards gelöst (max. {MAX_GUESSES} Guesses).")
    solved_numbers = [x for x in solution_guess_number if x is not None]
    if solved_numbers:
        print(f"Ø Lösungsversuch: {sum(solved_numbers) / len(solved_numbers):.3f} | "
              f"letztes Board: {max(solved_numbers)}")
    print(f"Guess-Target-Paare: {total_pair_evaluations:,}")

    return {
        "solved_count": solved_count,
        "solution_guess_number": solution_guess_number,
        "total_pair_evaluations": total_pair_evaluations,
    }


def benchmark(word_data, backend):
    """Mehrere zufällige Spiele simulieren und Statistik ausgeben."""
    games = BENCHMARK_GAMES
    results = []
    for game_no in range(1, games + 1):
        print(f"\n========== BENCHMARK-SPIEL {game_no}/{games} ==========")
        targets = load_targets(word_data.solution_words)
        results.append(play_game(word_data, targets, backend, game_id=game_no))

    attempts = [x for r in results for x in r["solution_guess_number"] if x is not None]
    print("\n" + "=" * 100 + "\nBENCHMARK\n" + "=" * 100)
    print(f"Spiele: {games} | Gelöst: {sum(r['solved_count'] for r in results)}"
          f"/{games * NUM_BOARDS}")
    if attempts:
        print(f"Ø {np.mean(attempts):.4f} | Median {np.median(attempts):.1f} | "
              f"P90 {np.percentile(attempts, 90):.1f} | Max {max(attempts)}")


def main():
    """Wortlisten laden und Simulation bzw. Benchmark starten."""
    solution_words, solution_counts, guess_words, guess_counts = load_word_data()
    word_data = WordData(solution_words, solution_counts, guess_words, guess_counts)
    backend = choose_backend()
    print(f"Backend: {backend} | wordfreq: {USE_WORDFREQ and zipf_frequency is not None}")

    if BENCHMARK_GAMES <= 1:
        play_game(word_data, load_targets(solution_words), backend, game_id=1)
    else:
        benchmark(word_data, backend)


if __name__ == "__main__":
    main()
