"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const SESSION_KEY = "little-boards-session-v1";
  const ui = { state: null, analysis: null, busy: false, analysisBusy: false, generation: 0, mode: "match", side: 1, hovered: null, info: null };
  const cells = new Map();
  const smallBoards = [];
  const location = (action) => ({ row: Math.floor(action / 9) + 1, column: action % 9 + 1, board: Math.floor(action / 27) * 3 + Math.floor((action % 9) / 3) + 1 });
  const percent = (rate) => Math.max(0, Math.min(100, Number(rate))).toFixed(1);
  const delta = (value) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)}%`;
  const isActive = () => ui.state && !ui.state.done && !ui.state.aborted;
  const isHumanTurn = () => isActive() && ui.state.to_play === ui.state.human_player;
  const matchingAnalysis = () => ui.analysis && ui.state && ui.analysis.session === ui.state.session && ui.analysis.revision === ui.state.revision;
  const adviceEnabled = () => ui.state && ui.state.mode === "coach" && isHumanTurn() && matchingAnalysis();

  function rememberSession(id) {
    try { id ? localStorage.setItem(SESSION_KEY, id) : localStorage.removeItem(SESSION_KEY); } catch (_) { /* Play also works without browser storage. */ }
  }

  function showError(message) {
    $("error-message").textContent = message;
    $("error-banner").hidden = false;
  }

  function clearError() { $("error-banner").hidden = true; }

  function handleExpiredSession(error) {
    if (error.status !== 404 && error.status !== 410) return false;
    ui.state = null;
    ui.analysis = null;
    ui.analysisBusy = false;
    ui.hovered = null;
    rememberSession(null);
    showError("The previous session is unavailable. Saved records remain on disk. Start a new game to continue.");
    return true;
  }

  async function api(path, payload) {
    let response;
    try {
      response = await fetch(path, { cache: "no-store", ...(payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }) });
    } catch (_) {
      throw new Error("The game server is unavailable. Check that it is running, then retry.");
    }
    let result;
    try { result = await response.json(); } catch (_) { throw new Error("The server returned an unreadable response. Please retry."); }
    if (!response.ok) {
      const error = new Error(result.error || "That action could not be completed. Please retry.");
      error.status = response.status;
      throw error;
    }
    return result;
  }

  function buildBoard() {
    for (let board = 0; board < 9; board++) {
      const group = document.createElement("div");
      group.className = "small-board";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", `Small board ${board + 1}`);
      for (let position = 0; position < 9; position++) {
        const action = (Math.floor(board / 3) * 3 + Math.floor(position / 3)) * 9 + (board % 3) * 3 + position % 3;
        const button = document.createElement("button");
        button.className = "cell empty";
        button.type = "button";
        button.dataset.action = String(action);
        button.addEventListener("click", () => playMove(action));
        button.addEventListener("mouseenter", () => previewMove(action));
        button.addEventListener("mouseleave", () => clearPreview(action));
        button.addEventListener("focus", () => previewMove(action));
        button.addEventListener("blur", () => clearPreview(action));
        button.addEventListener("keydown", (event) => navigateBoard(event, action));
        group.append(button);
        cells.set(action, button);
      }
      const claim = document.createElement("span");
      claim.className = "board-claim";
      claim.setAttribute("aria-hidden", "true");
      group.append(claim);
      smallBoards.push(group);
      $("board").append(group);
    }
  }

  function navigateBoard(event, action) {
    const steps = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -9, ArrowDown: 9 };
    if (!(event.key in steps)) return;
    event.preventDefault();
    const next = action + steps[event.key];
    if (next < 0 || next > 80 || (event.key === "ArrowLeft" && action % 9 === 0) || (event.key === "ArrowRight" && action % 9 === 8)) return;
    cells.get(next).focus();
  }

  function renderBoard() {
    const state = ui.state;
    const human = state ? state.human_player : ui.side;
    const legal = new Set(state && !state.aborted && !state.done ? state.legal_moves : []);
    const recommendations = new Map(adviceEnabled() ? ui.analysis.top_moves.map((move) => [move.action, move]) : []);
    const estimates = new Map(adviceEnabled() ? ui.analysis.moves.map((move) => [move.action, move]) : []);
    const availableBoards = new Set([...legal].map((action) => location(action).board - 1));
    for (let index = 0; index < 9; index++) {
      const status = state ? state.small_boards[index] : 0;
      const group = smallBoards[index];
      group.className = "small-board";
      if (availableBoards.has(index)) group.classList.add("available");
      if (status) group.classList.add("closed", status === 3 ? "drawn" : status === human ? "claimed-human" : "claimed-ai");
      group.querySelector(".board-claim").textContent = status === 1 ? "×" : status === 2 ? "○" : status === 3 ? "—" : "";
      group.setAttribute("aria-label", `Small board ${index + 1}${status ? status === 3 ? ", drawn and closed" : `, claimed by ${status === human ? "you" : "AI"}` : availableBoards.has(index) ? ", available" : ""}`);
    }
    for (const [action, button] of cells) {
      const value = state ? state.board[action] : 0;
      const canPlay = !ui.busy && isHumanTurn() && legal.has(action);
      const recommended = recommendations.get(action);
      const estimate = estimates.get(action);
      const coords = location(action);
      button.className = `cell ${value ? "occupied" : "empty"}`;
      button.setAttribute("aria-disabled", String(!canPlay));
      button.tabIndex = canPlay ? 0 : -1;
      button.replaceChildren();
      if (canPlay) button.classList.add("legal");
      if (state && action === state.last_move) button.classList.add("last-move");
      let label = `Row ${coords.row}, column ${coords.column}, small board ${coords.board}`;
      if (value) {
        const piece = document.createElement("span");
        piece.className = `piece ${value === human ? "human" : "ai"} ${value === 1 ? "cross" : "circle"}`;
        piece.setAttribute("aria-hidden", "true");
        button.append(piece);
        label += `, ${value === human ? "your" : "AI"} ${value === 1 ? "cross" : "circle"}`;
      } else if (recommended) {
        button.classList.add("recommended", `rank-${recommended.rank}`);
        const rank = document.createElement("span");
        rank.className = "rank-number";
        rank.textContent = String(recommended.rank);
        rank.setAttribute("aria-hidden", "true");
        button.append(rank);
        label += `, recommendation ${recommended.rank}`;
      } else { label += legal.has(action) ? ", legal move" : ", unavailable"; }
      if (estimate) label += `, estimated expected score ${percent(estimate.human_rate)}%, change ${delta(estimate.delta_pp)} percentage points`;
      button.setAttribute("aria-label", label);
      button.title = label;
    }
    $("board").classList.toggle("finished", Boolean(state && (state.done || state.aborted)));
    renderPreview();
  }

  function previewMove(action) {
    ui.hovered = action;
    renderPreview();
  }

  function clearPreview(action) {
    if (ui.hovered !== action) return;
    ui.hovered = null;
    renderPreview();
  }

  function renderPreview() {
    for (const [action, button] of cells) button.classList.toggle("previewed", action === ui.hovered && adviceEnabled() && ui.analysis.moves.some((move) => move.action === action));
    const estimate = adviceEnabled() && ui.analysis.moves.find((move) => move.action === ui.hovered);
    if (estimate) {
      const coords = location(estimate.action);
      $("preview-text").textContent = `Row ${coords.row} · Column ${coords.column} — expected score for you: ${percent(estimate.human_rate)}% (${delta(estimate.delta_pp)})`;
    } else if (isHumanTurn() && ui.state.mode === "coach") {
      $("preview-text").textContent = ui.analysisBusy ? "Calculating all legal moves… You may play while analysis runs." : matchingAnalysis() ? "Select or focus a legal move to view its estimate." : "Analysis is pending. You may still play.";
    } else if (isActive()) {
      $("preview-text").textContent = ui.state.next_board < 0 ? "Any legal square is available because the destination board is closed." : `${isHumanTurn() ? "Your" : "The AI’s"} next move must be in small board ${ui.state.next_board + 1}.`;
    } else {
      $("preview-text").textContent = "A small board closes when a player forms three in a row.";
    }
  }

  function renderHeader() {
    const state = ui.state;
    document.body.classList.toggle("has-session", Boolean(state));
    const human = state ? state.human_player : ui.side;
    $("human-symbol").textContent = human === 1 ? "×" : "○";
    $("ai-symbol").textContent = human === 1 ? "○" : "×";
    $("mode-label").textContent = state ? state.mode === "coach" ? "GUIDED ANALYSIS" : "MATCH" : "GAME SETUP";
    $("move-count").textContent = `MOVE ${String(state ? state.ply : 0).padStart(2, "0")}`;
    $("start-label").textContent = state ? "New game" : "Start game";
    $("start-button").disabled = ui.busy;
    $("abort-button").disabled = ui.busy || !isActive();
    $("pending-settings").hidden = !state || (state.mode === ui.mode && state.human_player === ui.side);
    document.querySelectorAll(".mode-button").forEach((button) => { button.classList.toggle("selected", button.dataset.mode === ui.mode); button.setAttribute("aria-pressed", String(button.dataset.mode === ui.mode)); button.disabled = ui.busy; });
    document.querySelectorAll(".side-button").forEach((button) => { button.classList.toggle("selected", Number(button.dataset.side) === ui.side); button.setAttribute("aria-pressed", String(Number(button.dataset.side) === ui.side)); button.disabled = ui.busy; });
    const thinking = Boolean(isActive() && state.to_play === state.ai_player);
    document.querySelector(".game-panel").classList.toggle("thinking", thinking && ui.busy);
    if (!state) {
      $("turn-title").textContent = ui.busy ? "Initializing game…" : "Select settings and start a game.";
      $("turn-hint").textContent = "Choose the mode and your player side.";
    } else if (state.aborted) {
      $("turn-title").textContent = "Game ended.";
      $("turn-hint").textContent = "The game ended before all nine small boards were closed. Start a new game to continue.";
    } else if (state.done) {
      $("turn-title").textContent = state.winner === 0 ? "Draw." : state.winner === human ? "You win." : "AI wins.";
      $("turn-hint").textContent = `Final small-board score: you ${state.score.human}, AI ${state.score.ai}. Start a new game to continue.`;
    } else if (thinking) {
      $("turn-title").textContent = ui.busy ? "AI is calculating…" : "AI turn.";
      $("turn-hint").textContent = "The model is evaluating legal moves.";
    } else {
      $("turn-title").textContent = "Your turn.";
      $("turn-hint").textContent = state.next_board < 0 ? "Select any legal square; all destination boards are closed." : `Select a legal square in small board ${state.next_board + 1} — board row ${Math.floor(state.next_board / 3) + 1}, board column ${state.next_board % 3 + 1}.`;
    }
    const score = state ? state.score : { human: 0, ai: 0 };
    $("board-score").replaceChildren(document.createTextNode("Small boards "));
    const scoreText = document.createElement("b");
    scoreText.textContent = `${score.human} : ${score.ai}`;
    $("board-score").append(scoreText);
  }

  function setRate(element, value) {
    element.replaceChildren(document.createTextNode(value));
    const suffix = document.createElement("small");
    suffix.textContent = "%";
    element.append(suffix);
  }

  function renderEvaluation() {
    const ready = matchingAnalysis();
    $("win-bar").classList.toggle("unavailable", !ready);
    $("analysis-status").className = `live-label${ui.analysisBusy ? " loading" : ready ? " active" : ""}`;
    $("analysis-status").textContent = ui.analysisBusy ? "ANALYZING" : ready ? ui.state.done ? "FINAL" : "LIVE" : "WAITING";
    if (ready) {
      const human = Number(percent(ui.analysis.human_rate));
      const ai = (100 - human).toFixed(1);
      setRate($("human-rate"), human.toFixed(1));
      setRate($("ai-rate"), ai);
      $("human-bar").style.width = `${human}%`;
      $("ai-bar").style.width = `${ai}%`;
      $("win-bar").setAttribute("aria-label", `Estimated expected score: human ${human.toFixed(1)} percent on the left, AI ${ai} percent on the right.`);
      $("evaluation-note").textContent = ui.state.done ? "Final expected-score estimate; draws count as 50%." : "Current model estimate for this position.";
    } else {
      setRate($("human-rate"), "—");
      setRate($("ai-rate"), "—");
      $("win-bar").setAttribute("aria-label", "Win rate has not been estimated for the current position.");
      $("evaluation-note").textContent = ui.state && ui.state.aborted ? "No final estimate: the game ended before completion." : ui.state ? "Waiting for analysis of the current position." : "Start a game to calculate the model estimate.";
    }
  }

  function emptyRecommendations(title, message, loading = false) {
    const container = $("recommendations");
    container.replaceChildren();
    if (loading) {
      for (let index = 0; index < 3; index++) { const block = document.createElement("div"); block.className = "skeleton-card"; block.setAttribute("aria-hidden", "true"); container.append(block); }
      const status = document.createElement("span");
      status.className = "section-description";
      status.textContent = message;
      container.append(status);
      return;
    }
    const empty = document.createElement("div");
    empty.className = "insight-empty";
    const mini = document.createElement("div");
    mini.className = "mini-board";
    mini.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 9; index++) mini.append(document.createElement("i"));
    const heading = document.createElement("strong"); heading.textContent = title;
    const copy = document.createElement("p"); copy.textContent = message;
    empty.append(mini, heading, copy);
    container.append(empty);
  }

  function renderRecommendations() {
    const state = ui.state;
    $("recommendation-description").textContent = "Highest-ranked legal moves from the current position.";
    if (!state || state.mode !== "coach") {
      emptyRecommendations("Guided analysis is off.", "Select Guided analysis to calculate and display the three highest-ranked legal moves.");
      return;
    }
    if (state.done || state.aborted) { emptyRecommendations("No active position.", "Start a new Guided analysis game to evaluate legal moves."); return; }
    if (!isHumanTurn()) { emptyRecommendations("AI turn in progress.", "Recommendations update after the AI move."); return; }
    if (!matchingAnalysis()) { emptyRecommendations("Analysis pending.", ui.analysisBusy ? "Calculating all legal moves…" : "Analysis is pending. You may still play.", ui.analysisBusy); return; }
    $("recommendation-description").textContent = "Select a move to play it.";
    $("recommendations").replaceChildren();
    for (const move of ui.analysis.top_moves) {
      const coords = location(move.action);
      const card = document.createElement("button");
      card.className = `move-card rank-${move.rank}`;
      card.type = "button";
      card.disabled = ui.busy;
      card.setAttribute("aria-label", `Play ranked move ${move.rank}, row ${coords.row}, column ${coords.column}. Estimated expected score ${percent(move.human_rate)} percent, change ${delta(move.delta_pp)}.`);
      const rank = document.createElement("span"); rank.className = "rank-badge"; rank.textContent = String(move.rank);
      const position = document.createElement("span");
      const coordinate = document.createElement("span"); coordinate.className = "move-position"; coordinate.textContent = `Row ${coords.row} · Col ${coords.column}`;
      const board = document.createElement("span"); board.className = "move-location"; board.textContent = `Small board ${coords.board}`;
      position.append(coordinate, board);
      const numbers = document.createElement("span"); numbers.className = "move-numbers";
      const change = document.createElement("span"); change.className = `move-delta${move.delta_pp < 0 ? " negative" : ""}`; change.textContent = delta(move.delta_pp);
      const rate = document.createElement("span"); rate.className = "move-rate"; rate.textContent = `${percent(move.human_rate)}% expected score`;
      numbers.append(change, rate);
      card.append(rank, position, numbers);
      card.addEventListener("click", () => playMove(move.action));
      card.addEventListener("mouseenter", () => previewMove(move.action));
      card.addEventListener("mouseleave", () => clearPreview(move.action));
      card.addEventListener("focus", () => previewMove(move.action));
      card.addEventListener("blur", () => clearPreview(move.action));
      $("recommendations").append(card);
    }
  }

  function renderRecord() {
    const state = ui.state;
    const history = state ? state.history : [];
    $("history-count").textContent = String(history.length);
    $("history").replaceChildren();
    if (!history.length) {
      const empty = document.createElement("p"); empty.className = "history-empty"; empty.textContent = "No moves recorded."; $("history").append(empty);
    }
    for (const move of [...history].reverse()) {
      const row = document.createElement("div"); row.className = "history-row";
      const number = document.createElement("span"); number.className = "history-ply"; number.textContent = String(move.ply).padStart(2, "0");
      const actor = document.createElement("span"); actor.className = "history-actor";
      const dot = document.createElement("i"); dot.className = `legend-dot ${move.actor === "human" ? "human" : "ai"}`;
      actor.append(dot, document.createTextNode(move.actor === "human" ? "You" : "AI"));
      const coords = location(move.action);
      const position = document.createElement("span"); position.className = "history-position"; position.textContent = `R${coords.row} · C${coords.column}`;
      row.append(number, actor, position); $("history").append(row);
    }
    const record = state && state.record;
    $("record-indicator").className = `record-indicator${record && record.saved ? " saved" : record && record.error ? " failed" : ""}`;
    $("record-status-text").textContent = record && record.error ? `Record save failed: ${record.error}` : record && record.saved ? isActive() ? `Autosaved · ${history.length} ${history.length === 1 ? "move" : "moves"}` : "Game record saved. Download it below." : isActive() ? "Saving current position…" : "Moves are saved automatically.";
    const available = Boolean(record && record.saved);
    const download = $("download-button");
    download.classList.toggle("disabled", !available);
    download.setAttribute("aria-disabled", String(!available));
    download.tabIndex = available ? 0 : -1;
    if (available) { download.href = `/api/record?session=${encodeURIComponent(state.session)}`; download.setAttribute("download", record.filename || "game-record.json"); }
    else { download.removeAttribute("href"); download.removeAttribute("download"); }
  }

  function render() { renderHeader(); renderBoard(); renderEvaluation(); renderRecommendations(); renderRecord(); }

  function acceptState(state) {
    ui.state = state;
    ui.analysis = null;
    ui.analysisBusy = false;
    ui.hovered = null;
    rememberSession(state.session);
  }

  async function analyze() {
    if (!ui.state || ui.state.aborted || ui.busy) return;
    const generation = ui.generation;
    const { session, revision } = ui.state;
    ui.analysisBusy = true;
    renderEvaluation(); renderRecommendations(); renderPreview();
    try {
      const result = await api("/api/analysis", { session, revision });
      if (generation !== ui.generation || !ui.state || session !== ui.state.session || revision !== ui.state.revision) return;
      if (result.session !== session || result.revision !== revision) return;
      ui.analysis = result;
    } catch (error) {
      if (generation === ui.generation && !handleExpiredSession(error)) showError(error.message);
    } finally {
      if (generation === ui.generation) { ui.analysisBusy = false; render(); }
    }
  }

  async function advance() {
    if (!ui.state || ui.busy || ui.state.aborted) return;
    if (!ui.state.done && ui.state.to_play === ui.state.ai_player) {
      await mutate("/api/ai", { session: ui.state.session, revision: ui.state.revision });
    } else { void analyze(); }
  }

  async function mutate(path, payload) {
    if (ui.busy) return;
    const generation = ++ui.generation;
    ui.busy = true;
    ui.analysisBusy = false;
    ui.analysis = null;
    ui.hovered = null;
    clearError();
    render();
    let accepted = false;
    try {
      const result = await api(path, payload);
      if (generation !== ui.generation) return;
      acceptState(result);
      accepted = true;
    } catch (error) {
      if (generation !== ui.generation) return;
      if (!handleExpiredSession(error)) showError(error.message);
      if (ui.state) {
        try {
          const latest = await api(`/api/state?session=${encodeURIComponent(ui.state.session)}`);
          if (generation === ui.generation) acceptState(latest);
        } catch (refreshError) { handleExpiredSession(refreshError); }
      }
    } finally {
      if (generation === ui.generation) { ui.busy = false; render(); }
    }
    if (accepted) void advance();
  }

  function playMove(action) {
    if (ui.busy || !isHumanTurn() || !ui.state.legal_moves.includes(action)) return;
    void mutate("/api/move", { session: ui.state.session, revision: ui.state.revision, action });
  }

  async function loadInfo() {
    const info = await api("/api/info");
    ui.info = info;
    const displayName = info.model.display_name || info.model.name || "Current champion";
    const version = info.model.version ? ` ${info.model.version}` : "";
    $("model-name").textContent = `${displayName}${version}`;
    const iteration = info.model.iteration === null || info.model.iteration === undefined ? "" : ` · Iteration ${info.model.iteration}`;
    const checkpoint = info.model.name ? `Checkpoint ${info.model.name} · ` : "";
    $("model-detail").textContent = `${checkpoint}${info.simulations} MCTS simulations${iteration}`;
  }

  async function restore() {
    let session;
    try { session = localStorage.getItem(SESSION_KEY); } catch (_) { return; }
    if (!session) return;
    const generation = ++ui.generation;
    ui.busy = true;
    render();
    try {
      const state = await api(`/api/state?session=${encodeURIComponent(session)}`);
      if (generation !== ui.generation) return;
      acceptState(state);
      ui.mode = state.mode;
      ui.side = state.human_player;
    } catch (error) {
      if (!handleExpiredSession(error)) showError(error.message);
    } finally {
      if (generation === ui.generation) { ui.busy = false; render(); }
    }
    if (ui.state) void advance();
  }

  document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => { if (ui.busy) return; ui.mode = button.dataset.mode; renderHeader(); }));
  document.querySelectorAll(".side-button").forEach((button) => button.addEventListener("click", () => { if (ui.busy) return; ui.side = Number(button.dataset.side); renderHeader(); }));
  $("start-button").addEventListener("click", () => { void mutate("/api/new", { mode: ui.mode, human_player: ui.side, ...(ui.state ? { previous_session: ui.state.session } : {}) }); });
  $("abort-button").addEventListener("click", () => { if (isActive()) void mutate("/api/abort", { session: ui.state.session, revision: ui.state.revision }); });
  $("dismiss-error").addEventListener("click", clearError);
  $("retry-button").addEventListener("click", async () => {
    if (ui.busy) return;
    clearError();
    try { if (!ui.info) await loadInfo(); } catch (error) { showError(error.message); return; }
    if (ui.state) {
      const generation = ++ui.generation;
      ui.busy = true; render();
      try { const state = await api(`/api/state?session=${encodeURIComponent(ui.state.session)}`); if (generation === ui.generation) acceptState(state); }
      catch (error) { if (!handleExpiredSession(error)) showError(error.message); }
      finally { if (generation === ui.generation) { ui.busy = false; render(); } }
      void advance();
    } else { void restore(); }
  });
  buildBoard();
  render();
  void loadInfo().catch((error) => { $("model-name").textContent = "Model connection unavailable"; $("model-detail").textContent = "Start the local server, then retry."; showError(error.message); });
  void restore();
})();
