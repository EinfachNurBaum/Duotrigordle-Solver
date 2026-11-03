// ==UserScript==
// @name         Duotrigordle Solver Bridge
// @namespace    duo-bridge.local
// @version      1.1.0
// @description  Spielt Duotrigordle mit einem lokalen Python-Solver: liest die Boards, fragt den Solver, tippt den Guess.
// @match        https://duotrigordle.com/*
// @match        https://www.duotrigordle.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// @grant        unsafeWindow
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-idle
// @noframes
// ==/UserScript==

(function () {
  "use strict";

  // ---------- Konfiguration ----------
  const CFG = {
    SERVER: "http://127.0.0.1:8765",
    NUM_BOARDS: 32,
    MAX_GUESSES: 37,
    INPUT_METHOD: "auto", // "auto" | "keyboard" | "click"
    KEY_DELAY_MS: 40,
    SUBMIT_TIMEOUT_MS: 6000,
    PAUSE_BETWEEN_GUESSES_MS: 400,
    COLOR_CLASS: { G: /^_?green(_|$)/i, Y: /^_?yellow(_|$)/i, B: /^_?black(_|$)/i },
    CELL_SELECTOR: '[class*="_cell_"]',
    LETTER_SELECTOR: '[class*="_letter_"]',
    GHOST_CLASS: /textGhost/i,
    KEYBOARD_SELECTOR: '[class*="_keyboard_"] button',
  };

  let running = false;
  let stepOnly = false;
  let inputMethod = CFG.INPUT_METHOD === "auto" ? "keyboard" : CFG.INPUT_METHOD;
  let exclude = loadExclude();

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------- DOM lesen ----------

  /** Zellen nach Elternelement gruppieren: jedes Elternelement = ein Board. */
  function findBoards() {
    const groups = new Map();
    for (const cell of document.querySelectorAll(CFG.CELL_SELECTOR)) {
      const parent = cell.parentElement;
      if (!groups.has(parent)) groups.set(parent, []);
      groups.get(parent).push(cell);
    }
    return [...groups.entries()]
      .filter(([, cells]) => cells.length >= 5)
      .map(([el, cells]) => ({ el, cells }));
  }

  /** Farbe einer Zelle: "G", "Y", "." oder null (ungefärbt). */
  function cellColor(cell) {
    for (const cls of cell.classList) {
      if (CFG.COLOR_CLASS.G.test(cls)) return "G";
      if (CFG.COLOR_CLASS.Y.test(cls)) return "Y";
      if (CFG.COLOR_CLASS.B.test(cls)) return ".";
    }
    return null;
  }

  /** Buchstabe einer Zelle in Kleinschreibung. */
  function cellLetter(cell) {
    const span = cell.querySelector(CFG.LETTER_SELECTOR) || cell;
    return (span.textContent || "").trim().toLowerCase();
  }

  /** True für ausgegraute Hinweisbuchstaben der Seite. */
  function isGhost(cell) {
    const nodes = [cell, ...cell.querySelectorAll("*")];
    return nodes.some((n) => [...n.classList].some((c) => CFG.GHOST_CLASS.test(c)));
  }

  /** Alle gefärbten Zeilen eines Boards als [{word, pattern}]. */
  function readColoredRows(board) {
    const rows = [];
    for (let i = 0; i + 5 <= board.cells.length; i += 5) {
      const cells = board.cells.slice(i, i + 5);
      const colors = cells.map(cellColor);
      if (colors.some((c) => c === null)) break;
      rows.push({ word: cells.map(cellLetter).join(""), pattern: colors.join("") });
    }
    return rows;
  }

  /** Aktuell eingetippte Buchstaben aus der Eingabezeile eines ungelösten Boards. */
  function readInput(boards) {
    for (const board of boards) {
      const rows = readColoredRows(board);
      if (rows.length && rows[rows.length - 1].pattern === "GGGGG") continue;
      const n = rows.length * 5;
      const cells = board.cells.slice(n, n + 5);
      if (cells.length < 5) continue;
      if (cells.some((c) => cellColor(c) !== null)) continue;
      return cells.map((c) => (isGhost(c) ? "" : cellLetter(c))).join("");
    }
    return "";
  }

  /** Spielverlauf lesen: history[k] = {guess, feedback: 32 x "GY..." oder null für bereits gelöst}. */
  function readState() {
    const boards = findBoards();
    if (boards.length !== CFG.NUM_BOARDS) {
      throw new Error(
        `${boards.length} Boards gefunden statt ${CFG.NUM_BOARDS}. ` +
          `Ist ein normales Spiel geöffnet? (DuoBridge.dump() zur Diagnose)`
      );
    }
    const rowsPer = boards.map(readColoredRows);
    const len = Math.max(...rowsPer.map((r) => r.length));
    const history = [];
    for (let k = 0; k < len; k++) {
      const words = new Set(rowsPer.filter((r) => r.length > k).map((r) => r[k].word));
      if (words.size !== 1) {
        throw new Error(`Zeile ${k + 1}: Boards zeigen verschiedene Wörter: ${[...words].join(", ")}`);
      }
      history.push({
        guess: [...words][0],
        feedback: rowsPer.map((r) => (r.length > k ? r[k].pattern : null)),
      });
    }
    const solved = rowsPer.map((r) => r.length > 0 && r[r.length - 1].pattern === "GGGGG");
    return {
      boards,
      history,
      solvedCount: solved.filter(Boolean).length,
      done: solved.every(Boolean) || history.length >= CFG.MAX_GUESSES,
    };
  }

  // ---------- Eingabe ----------

  /** Button der Bildschirmtastatur zu einer Taste finden. */
  function findKeyButton(key) {
    const label = key === "Enter" ? "⏎" : key === "Backspace" ? "⌫" : key.toUpperCase();
    return [...document.querySelectorAll(CFG.KEYBOARD_SELECTOR)].find(
      (b) => (b.firstElementChild?.textContent || b.textContent).trim() === label
    );
  }

  /** Eine Taste per keydown-Event oder Klick auslösen. */
  async function pressKey(key) {
    if (inputMethod === "keyboard") {
      const opts = { key, code: key.length === 1 ? "Key" + key.toUpperCase() : key, bubbles: true, cancelable: true };
      window.dispatchEvent(new KeyboardEvent("keydown", opts));
      window.dispatchEvent(new KeyboardEvent("keyup", opts));
    } else {
      const btn = findKeyButton(key);
      if (!btn) throw new Error(`Taste "${key}" auf der Bildschirmtastatur nicht gefunden`);
      btn.click();
    }
    await sleep(CFG.KEY_DELAY_MS);
  }

  /** Eingabezeile mit Backspace leeren. */
  async function clearInput() {
    for (let i = 0; i < 6 && readInput(findBoards()) !== ""; i++) await pressKey("Backspace");
  }

  /** Wort eintippen und prüfen; bei Bedarf auf Klick-Eingabe wechseln. */
  async function typeWord(word) {
    for (let attempt = 0; attempt < 2; attempt++) {
      await clearInput();
      for (const ch of word.toLowerCase()) await pressKey(ch);
      await sleep(80);
      if (readInput(findBoards()) === word.toLowerCase()) return;
      if (CFG.INPUT_METHOD === "auto" && inputMethod === "keyboard") {
        log("Tastatur-Events wirken nicht – wechsle auf Klicks der Bildschirmtastatur.");
        inputMethod = "click";
        continue;
      }
      break;
    }
    throw new Error(`Eingabe von "${word}" nicht im Board sichtbar (gelesen: "${readInput(findBoards())}")`);
  }

  /** Enter drücken und auf eine neue gefärbte Zeile warten. */
  async function submit(prevLen) {
    await pressKey("Enter");
    const t0 = Date.now();
    while (Date.now() - t0 < CFG.SUBMIT_TIMEOUT_MS) {
      await sleep(100);
      try {
        if (readState().history.length > prevLen) return true;
      } catch (_) { /* Zwischenzustand während des Renderns */ }
    }
    return false;
  }

  // ---------- Server ----------

  /** JSON-Anfrage an den lokalen Bridge-Server. */
  function api(method, path, body) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method,
        url: CFG.SERVER + path,
        headers: { "Content-Type": "application/json" },
        data: body ? JSON.stringify(body) : undefined,
        timeout: 0,
        onload: (r) => {
          let json;
          try { json = JSON.parse(r.responseText); } catch (_) { json = { error: r.responseText }; }
          if (r.status >= 200 && r.status < 300) resolve(json);
          else reject(new Error(json.error || `HTTP ${r.status}`));
        },
        onerror: () => reject(new Error(`Server ${CFG.SERVER} nicht erreichbar – läuft duo_bridge_server.py?`)),
      });
    });
  }

  /** Spiel-ID aus Datum und Spielmodus. */
  function gameId() {
    const mode = location.pathname.replace(/\W+/g, "-").replace(/^-|-$/g, "") || "daily";
    return `${new Date().toISOString().slice(0, 10)}_${mode}`;
  }

  /** Liste abgelehnter Wörter laden. */
  function loadExclude() {
    try { return JSON.parse(GM_getValue("exclude", "[]")); } catch (_) { return []; }
  }
  /** Liste abgelehnter Wörter speichern. */
  function saveExclude() { GM_setValue("exclude", JSON.stringify(exclude)); }

  // ---------- Spielschleife ----------

  /** Einen Zug spielen; liefert "played", "rejected" oder "done". */
  async function step() {
    const state = readState();
    if (state.done) {
      await finish(state);
      return "done";
    }
    setStatus(`Guess ${state.history.length + 1}: Solver rechnet … (${state.solvedCount}/32 gelöst)`);
    const res = await api("POST", "/next", { history: state.history, exclude, game_id: gameId() });
    if (res.done) { await finish(state); return "done"; }

    showDecision(res);
    if (!running && !stepOnly) return "played";

    await typeWord(res.guess);
    const ok = await submit(state.history.length);
    if (!ok) {
      log(`"${res.guess.toUpperCase()}" wurde nicht angenommen → ausgeschlossen.`);
      exclude.push(res.guess);
      saveExclude();
      await clearInput();
      return "rejected";
    }
    return "played";
  }

  /** Spielende an den Server melden. */
  async function finish(state) {
    try {
      const r = await api("POST", "/finish", { history: state.history, exclude, game_id: gameId(), url: location.href });
      setStatus(`Fertig: ${r.solved}/32 in ${r.guesses_used} Guesses (Ø ${r.avg_solve_guess})`);
    } catch (e) {
      setStatus(`Fertig (Log fehlgeschlagen: ${e.message})`);
    }
  }

  /** Züge spielen, bis das Spiel vorbei ist oder gestoppt wird. */
  async function run() {
    if (running) return;
    running = true;
    updateButtons();
    try {
      while (running) {
        const r = await step();
        if (r === "done") break;
        await sleep(CFG.PAUSE_BETWEEN_GUESSES_MS);
      }
    } catch (e) {
      setStatus("Fehler: " + e.message);
      console.error("[DuoBridge]", e);
    } finally {
      running = false;
      updateButtons();
    }
  }

  /** Genau einen Zug spielen. */
  async function runOnce() {
    if (running) return;
    running = true;
    stepOnly = true;
    updateButtons();
    try { await step(); setStatus("Schritt fertig."); }
    catch (e) { setStatus("Fehler: " + e.message); console.error("[DuoBridge]", e); }
    finally { running = false; stepOnly = false; updateButtons(); }
  }

  /** Nach dem laufenden Zug anhalten. */
  function stop() {
    running = false;
    setStatus("Stoppt nach dem aktuellen Zug …");
  }

  // ---------- Panel ----------
  const panel = document.createElement("div");
  panel.id = "duo-bridge-panel";
  panel.innerHTML = `
    <style>
      #duo-bridge-panel{position:fixed;right:12px;bottom:12px;z-index:2147483647;width:300px;
        font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;background:#111c;color:#eee;
        border:1px solid #555;border-radius:8px;padding:8px;backdrop-filter:blur(4px)}
      #duo-bridge-panel button{font:inherit;margin-right:4px;padding:2px 8px;border-radius:4px;
        border:1px solid #777;background:#333;color:#eee;cursor:pointer}
      #duo-bridge-panel button:disabled{opacity:.4;cursor:default}
      #duo-bridge-panel .st{margin:6px 0;white-space:pre-wrap}
      #duo-bridge-panel .log{max-height:140px;overflow:auto;border-top:1px solid #444;padding-top:4px;opacity:.85}
    </style>
    <div><b>Solver Bridge</b>
      <span style="float:right"><button data-a="min" tabindex="-1">–</button></span></div>
    <div class="body">
      <div style="margin-top:6px">
        <button data-a="run" tabindex="-1">▶ Start</button>
        <button data-a="step" tabindex="-1">⏭ Schritt</button>
        <button data-a="stop" tabindex="-1">■ Stop</button>
      </div>
      <div class="st">Bereit.</div>
      <div class="log"></div>
    </div>`;
  document.body.appendChild(panel);
  const $ = (s) => panel.querySelector(s);

  panel.addEventListener("mousedown", (e) => e.preventDefault());
  panel.addEventListener("click", (e) => {
    const a = e.target?.dataset?.a;
    e.stopPropagation();
    if (a === "run") run();
    else if (a === "step") runOnce();
    else if (a === "stop") stop();
    else if (a === "min") $(".body").style.display = $(".body").style.display === "none" ? "" : "none";
  });

  /** Statuszeile setzen. */
  function setStatus(t) { $(".st").textContent = t; }
  /** Zeile ins Panel-Log und in die Konsole schreiben. */
  function log(t) {
    const d = document.createElement("div");
    d.textContent = t;
    $(".log").prepend(d);
    console.log("[DuoBridge]", t);
  }
  /** Buttons je nach Laufzustand aktivieren. */
  function updateButtons() {
    $('[data-a="run"]').disabled = running;
    $('[data-a="step"]').disabled = running;
    $('[data-a="stop"]').disabled = !running;
  }
  /** Entscheidung des Solvers im Log anzeigen. */
  function showDecision(r) {
    log(
      `#${r.guess_number} ${r.guess.toUpperCase()} [${r.candidate ? "CAND" : "PROBE"}] ` +
        `${r.reason} ELog=${r.ELog.toFixed(2)} N=${r.N} S=${r.SingletonCount} ` +
        `${r.cached ? "(cache)" : r.seconds + "s"}`
    );
  }
  updateButtons();

  // ---------- Konsole und Menü ----------
  const DuoBridge = {
    run, runOnce, stop, readState,
    /** Diagnose: Zellklassen, gelesener Zustand und Eingabemethode ausgeben. */
    dump() {
      const boards = findBoards();
      console.log("Boards gefunden:", boards.length);
      if (boards[0]) {
        console.log("Klassen Board 1, erste 10 Zellen:",
          boards[0].cells.slice(0, 10).map((c) => c.className));
      }
      try { console.log("Zustand:", readState()); } catch (e) { console.warn(e.message); }
      console.log("Eingabe:", JSON.stringify(readInput(boards)), "Methode:", inputMethod);
    },
    /** Liste abgelehnter Wörter leeren. */
    clearExclude() { exclude = []; saveExclude(); log("Ausschlussliste geleert."); },
  };
  try { (typeof unsafeWindow !== "undefined" ? unsafeWindow : window).DuoBridge = DuoBridge; } catch (_) {}

  GM_registerMenuCommand("Solver: Start", run);
  GM_registerMenuCommand("Solver: Ein Schritt", runOnce);
  GM_registerMenuCommand("Solver: Diagnose (Konsole)", DuoBridge.dump);
  GM_registerMenuCommand("Solver: Ausschlussliste leeren", DuoBridge.clearExclude);

  api("GET", "/health")
    .then((h) => log(`Server ok (${h.backend}, ${h.fingerprint})`))
    .catch((e) => log(e.message));
})();
