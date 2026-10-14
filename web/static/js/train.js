// -------------------------------------------------------------
// Live Training Studio & Telemetry Engine
// -------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const ui = {
  studioWrapper: $("studioWrapper"),
  studioStatus: $("studioStatus"),
  btnCloseStudio: $("btnCloseStudio"),
  btnToggleRetrain: $("btnToggleRetrain"),
  btnStartTrain: $("btnStartTrain"),
  btnPauseTrain: $("btnPauseTrain"),
  btnSaveTrain: $("btnSaveTrain"),
  btnStopTrain: $("btnStopTrain"),
  inpEpochs: $("inpEpochs"),
  inpLR: $("inpLR"),
  inpBatch: $("inpBatch"),
  inpVerbose: $("inpVerbose"),
  trainProgFill: $("trainProgFill"),
  progBarTrack: document.querySelector(".prog-bar-track"),
  telEpoch: $("telEpoch"),
  telBatch: $("telBatch"),
  telLoss: $("telLoss"),
  telBatchAcc: $("telBatchAcc"),
  telTestAcc: $("telTestAcc"),
  telSpeed: $("telSpeed"),
  telTime: $("telTime"),
  telSamples: $("telSamples"),
  trainConsole: $("trainConsole"),
  lossCanvas: $("lossCanvas"),
  accCanvas: $("accCanvas"),
  lossHeadVal: $("lossHeadVal"),
  accHeadVal: $("accHeadVal"),
  statAccuracy: $("statAccuracy"),
  statAccuracyPreview: $("statAccuracyPreview"),
  statLoss: $("statLoss"),
  statEpochs: $("statEpochs"),
  statSteps: $("statSteps"),
  statDuration: $("statDuration"),
  footerDuration: $("footerDuration"),
  statsCollapsible: $("statsCollapsible"),
  statToggleIcon: $("statToggleIcon"),
  btnToggleStats: $("btnToggleStats"),
  telemetryDetails: $("telemetryDetails"),
  telToggleIcon: $("telToggleIcon"),
  telToggleText: $("telToggleText"),
  btnToggleTelemetry: $("btnToggleTelemetry"),
};

let trainEventSource = null;
let isPanelOpen = false;
let isPaused = false;
let lossHistory = [];
let accHistory = [];
let totalSteps = 0;
let completedEpochs = 0;
let renderRafId = null;

// Reusable Collapsible Panel Helper
function togglePanel({
  panel,
  trigger,
  icon,
  textEl,
  openText,
  closedText,
  activeClass = "active",
}) {
  if (!panel) return false;
  const isCurrentlyHidden = panel.classList.contains("is-hidden");
  panel.classList.toggle("is-hidden", !isCurrentlyHidden);
  const isOpen = isCurrentlyHidden;
  if (trigger) trigger.classList.toggle(activeClass, isOpen);
  if (icon) icon.textContent = isOpen ? "\u25BE" : "\u25B8";
  if (textEl && openText && closedText) {
    textEl.textContent = isOpen ? openText : closedText;
  }
  return isOpen;
}

function toggleRetrainPanel() {
  if (ui.btnCloseStudio.disabled) return;
  isPanelOpen = togglePanel({
    panel: ui.studioWrapper,
    trigger: ui.btnToggleRetrain,
  });
  if (isPanelOpen) {
    requestAnimationFrame(() => {
      renderGraphs();
      ui.studioWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  } else if (trainEventSource) {
    stopLiveTraining();
  }
}

function toggleStatsCollapse() {
  togglePanel({
    panel: ui.statsCollapsible,
    trigger: ui.btnToggleStats,
    icon: ui.statToggleIcon,
  });
}

function toggleTelemetryDetails() {
  togglePanel({
    panel: ui.telemetryDetails,
    trigger: ui.btnToggleTelemetry,
    icon: ui.telToggleIcon,
    textEl: ui.telToggleText,
    openText: "less",
    closedText: "details",
  });
}

const MAX_CONSOLE_NODES = 120;
const clearConsole = () => {
  ui.trainConsole.innerHTML = '<span class="term-dim">// output here</span>\n';
};
const pad = (val, len) => String(val).padStart(len, " ");
const fmtSec = (sec) => sec.toFixed(2).padStart(6, " ") + "S";

function appendConsoleLine(html) {
  ui.trainConsole.insertAdjacentHTML("beforeend", html);
  while (ui.trainConsole.childNodes.length > MAX_CONSOLE_NODES) {
    ui.trainConsole.removeChild(ui.trainConsole.firstChild);
  }
  ui.trainConsole.scrollTop = ui.trainConsole.scrollHeight;
}

function prepareCanvas(canvas) {
  const dpr = Math.max(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const tw = Math.round(w * dpr);
  const th = Math.round(h * dpr);
  if (canvas.width !== tw || canvas.height !== th) {
    canvas.width = tw;
    canvas.height = th;
  }
  const ctx = canvas.getContext("2d");
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return { ctx, w, h };
}

function drawChart(
  canvas,
  history,
  strokeColor,
  fillRgba,
  fmtFn,
  boundsFn,
  headEl,
) {
  const { ctx, w, h } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const padLeft = 64,
    padRight = 18,
    padTop = 18,
    padBottom = 24;
  const pw = w - padLeft - padRight;
  const ph = h - padTop - padBottom;
  const N = history.length;
  const { floor, ceil } = boundsFn(history);
  const span = Math.max(ceil - floor, 0.0001);

  // Gridlines & Y-axis labels
  ctx.font = "600 12.5px 'JetBrains Mono', monospace";
  for (const f of [0, 0.25, 0.5, 0.75, 1.0]) {
    const y = Math.round(padTop + (1 - f) * ph) + 0.5;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + pw, y);
    ctx.stroke();

    ctx.fillStyle = strokeColor;
    ctx.textAlign = "right";
    ctx.fillText(fmtFn(floor + f * span), padLeft - 8, y + 4.5);
  }

  // Step Axis
  ctx.font = "12px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#888";
  ctx.textAlign = "left";
  ctx.fillText("Step 1", padLeft, h - 7);
  ctx.textAlign = "right";
  ctx.fillText(`Step ${totalSteps || N || 1}`, padLeft + pw, h - 7);

  if (N === 0) return;

  const xSpan = Math.max(totalSteps ? totalSteps - 1 : N - 1, 1);
  const getY = (v) => padTop + (1 - (v - floor) / span) * ph;
  const outerGlow =
    strokeColor === "#f59e0b"
      ? "rgba(245, 158, 11, 0.35)"
      : "rgba(34, 197, 94, 0.35)";

  if (N === 1) {
    const x = padLeft,
      y = getY(history[0]);
    ctx.fillStyle = outerGlow;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = strokeColor;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
    headEl.textContent = fmtFn(history[0]);
    return;
  }

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, padTop, 0, padTop + ph);
  grad.addColorStop(0, fillRgba);
  grad.addColorStop(1, "rgba(0, 0, 0, 0)");

  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = padLeft + (i / xSpan) * pw;
    const y = getY(history[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.0;
  ctx.stroke();

  ctx.lineTo(padLeft + ((N - 1) / xSpan) * pw, padTop + ph);
  ctx.lineTo(padLeft, padTop + ph);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Active glowing tip
  const lastX = padLeft + ((N - 1) / xSpan) * pw;
  const lastY = getY(history[N - 1]);
  ctx.fillStyle = outerGlow;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = strokeColor;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, 2 * Math.PI);
  ctx.fill();

  headEl.textContent = fmtFn(history[N - 1]);
}

function lossBounds(hist) {
  if (hist.length === 0) return { floor: 0.0, ceil: 2.5 };
  const minL = Math.min(...hist),
    maxL = Math.max(...hist);
  const span = Math.max(maxL - minL, 0.15);
  return { floor: Math.max(0, minL - span * 0.08), ceil: maxL + span * 0.08 };
}

function accBounds(hist) {
  if (hist.length === 0) return { floor: 0, ceil: 100 };
  const minA = Math.min(...hist),
    maxA = Math.max(...hist);
  let floor = Math.max(0, Math.floor((minA - 4) / 5) * 5);
  let ceil = Math.min(100, Math.ceil((maxA + 4) / 5) * 5);
  if (ceil - floor < 20) {
    ceil = Math.min(100, floor + 20);
    floor = Math.max(0, ceil - 20);
  }
  return { floor, ceil };
}

function renderGraphs() {
  drawChart(
    ui.lossCanvas,
    lossHistory,
    "#f59e0b",
    "rgba(245, 158, 11, 0.25)",
    (v) => v.toFixed(4),
    lossBounds,
    ui.lossHeadVal,
  );
  drawChart(
    ui.accCanvas,
    accHistory,
    "#22c55e",
    "rgba(34, 197, 94, 0.22)",
    (v) => `${v.toFixed(2)}%`,
    accBounds,
    ui.accHeadVal,
  );
}

function scheduleRenderGraphs() {
  if (renderRafId) return;
  renderRafId = requestAnimationFrame(() => {
    renderGraphs();
    renderRafId = null;
  });
}

function updateStatsUI(s) {
  const has = s.epochs > 0;
  ui.statAccuracy.textContent = ui.statAccuracyPreview.textContent = has
    ? s.accuracy.toFixed(2) + "%"
    : "--%";
  ui.statLoss.textContent = has ? s.loss.toFixed(4) : "--";
  ui.statEpochs.textContent = has ? s.epochs : "0";
  ui.statSteps.textContent = has ? s.steps.toLocaleString() : "--";
  ui.statDuration.textContent = has
    ? s.training_duration_sec.toFixed(4) + "s"
    : "--";
  if (has && ui.footerDuration) {
    ui.footerDuration.innerHTML = `&tau;<sub>train</sub>: ${s.training_duration_sec.toFixed(4)}s`;
  }
}

// Concise Data-Driven Studio UI State Machine
function syncStudioUI(state) {
  const canControl = completedEpochs >= 1;
  const isRunning = state === "running";
  const isPaused = state === "paused";
  const isActive = isRunning || isPaused;

  ui.btnStartTrain.disabled = isActive;
  ui.btnCloseStudio.disabled = isActive;
  ui.btnCloseStudio.classList.toggle("disabled", isActive);

  ui.btnPauseTrain.disabled = !isActive || !canControl;
  ui.btnPauseTrain.textContent = isPaused ? "resume" : "pause";
  ui.btnPauseTrain.title = !isActive
    ? state === "idle"
      ? "Pause available after 1st epoch completed"
      : ""
    : isPaused
      ? "Resume live training"
      : canControl
        ? "Pause live training"
        : "Pause available after 1st epoch completed";

  ui.btnStopTrain.disabled = !isActive || !canControl;
  ui.btnStopTrain.title = !isActive
    ? state === "idle"
      ? "Stop available after 1st epoch completed"
      : ""
    : canControl
      ? "Stop live training"
      : "Stop available after 1st epoch completed";

  const showSave = isPaused && canControl;
  ui.btnSaveTrain.classList.toggle("is-hidden", !showSave);
  ui.btnSaveTrain.disabled = !showSave;
  ui.btnSaveTrain.title = canControl
    ? `Save epoch ${completedEpochs} checkpoint`
    : "Save available after 1st epoch completed";
  ui.btnSaveTrain.textContent = canControl ? "save" : "save (ep \u2265 1)";

  const badgeMap = {
    idle: ["phase-idle", "idle"],
    running: ["phase-forward", "training live..."],
    paused: ["phase-paused", "paused"],
    completed: ["phase-forward", "completed"],
    stopped: [
      "phase-idle",
      completedEpochs > 0 ? `stopped (saved ep ${completedEpochs})` : "stopped",
    ],
  };
  const [badgeCls, badgeTxt] = badgeMap[state] || badgeMap.idle;
  ui.studioStatus.className = `phase-badge ${badgeCls}`;
  ui.studioStatus.textContent = badgeTxt;

  ui.progBarTrack.classList.toggle("is-hidden", !isActive);
  if (state === "completed") ui.trainProgFill.style.width = "100%";
}

function startLiveTraining() {
  const epochs = parseInt(ui.inpEpochs.value) || 10;
  const lr = parseFloat(ui.inpLR.value) || 0.1;
  const batchSize = parseInt(ui.inpBatch.value) || 64;
  const verbose = parseInt(ui.inpVerbose.value);

  completedEpochs = 0;
  isPaused = false;
  ui.trainProgFill.style.width = "0%";
  syncStudioUI("running");

  updateStatsUI({
    accuracy: 0,
    loss: 0,
    epochs: 0,
    steps: 0,
    training_duration_sec: 0,
  });
  ui.lossHeadVal.textContent = "--";
  ui.accHeadVal.textContent = "--%";
  ui.telTime.textContent = "Time: 0.0s";

  lossHistory = [];
  accHistory = [];
  renderGraphs();
  clearConsole();

  totalSteps = 0;

  trainEventSource = new EventSource(
    `/train/start?epochs=${epochs}&lr=${lr}&batch_size=${batchSize}`,
  );

  trainEventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === "start") {
      totalSteps = data.total_steps;
      ui.telEpoch.textContent = `Epoch: 1/${data.total_epochs}`;
      ui.telBatch.textContent = `Batch: 1/${data.total_batches}`;
      ui.telSamples.textContent = `${data.train_samples.toLocaleString()} train / ${data.test_samples.toLocaleString()} test`;

      if (verbose >= 1) {
        appendConsoleLine(
          `<span class="term-red">[  0.00S] STRT // SMPLS: ${pad(data.train_samples, 6)} // EPOC: ${pad(data.total_epochs, 6)} // BTCH: ${pad(batchSize, 6)}</span>\n\n`,
        );
      }
      if (verbose >= 2) {
        appendConsoleLine(
          `<span class="term-yellow">    [  0.00S] EPOC ${pad(0, 6)}/${pad(data.total_epochs, 6)} | LOSS: ${pad("-", 6)} | TACC: ${pad("-", 6)}%</span>\n`,
        );
      }
    } else if (data.type === "step") {
      lossHistory.push(data.loss);
      accHistory.push(data.accuracy);
      ui.trainProgFill.style.width =
        Math.min(100, (data.step / data.total_steps) * 100) + "%";
      ui.telEpoch.textContent = `Epoch: ${data.epoch}/${data.total_epochs}`;
      ui.telBatch.textContent = `Batch: ${data.batch}/${data.total_batches}`;
      ui.telLoss.textContent = `Loss: ${data.loss.toFixed(4)}`;
      ui.telBatchAcc.textContent = `Batch Acc: ${data.accuracy.toFixed(2)}%`;
      ui.telTime.textContent = `Time: ${data.elapsed_sec.toFixed(1)}s`;
      ui.telSpeed.textContent = `Speed: ${data.samples_per_sec.toLocaleString()} smp/s`;

      if (verbose >= 3) {
        appendConsoleLine(
          `<span class="term-dim">       [${fmtSec(data.elapsed_sec)}] STEP ${pad(data.step, 6)} (BTCH ${pad(data.batch, 6)}) | LOSS: ${data.loss.toFixed(4).padStart(6, " ")} | BACC: ${data.accuracy.toFixed(2).padStart(6, " ")}%</span>\n`,
        );
      }
      scheduleRenderGraphs();
    } else if (data.type === "epoch") {
      completedEpochs = data.epoch;
      ui.telTestAcc.textContent = `Test Acc: ${data.accuracy.toFixed(2)}%`;
      ui.telEpoch.textContent = `Epoch: ${data.epoch < data.total_epochs ? data.epoch + 1 : data.epoch}/${data.total_epochs}`;
      ui.telTime.textContent = `Time: ${data.elapsed_sec.toFixed(1)}s`;

      syncStudioUI(isPaused ? "paused" : "running");

      updateStatsUI({
        accuracy: data.accuracy,
        loss: data.loss,
        epochs: data.epoch,
        steps: data.step,
        training_duration_sec: data.elapsed_sec,
      });

      if (verbose >= 2) {
        appendConsoleLine(
          `<span class="term-yellow">    [${fmtSec(data.elapsed_sec)}] EPOC ${pad(data.epoch, 6)}/${pad(data.total_epochs, 6)} | LOSS: ${data.loss.toFixed(4).padStart(6, " ")} | TACC: ${data.accuracy.toFixed(2).padStart(6, " ")}%</span>\n`,
        );
      }
      renderGraphs();
    } else if (data.type === "end") {
      updateStatsUI({
        accuracy: data.accuracy,
        loss: data.loss,
        epochs: data.epoch,
        steps: data.step,
        training_duration_sec: data.elapsed_sec,
      });
      ui.telTime.textContent = `Time: ${data.elapsed_sec.toFixed(2)}s`;
      stopLiveTraining(true);
      syncStudioUI("completed");

      if (verbose >= 1) {
        appendConsoleLine(
          `\n<span class="term-red">[${fmtSec(data.elapsed_sec)}] OVER // FACC: ${data.accuracy.toFixed(2).padStart(6, " ")}% // FLOS: ${data.loss.toFixed(4).padStart(6, " ")} // DURN: ${fmtSec(data.elapsed_sec)}</span>\n`,
        );
      }
    }
  };

  trainEventSource.onerror = () => stopLiveTraining(true);
}

async function handleTrainAction(action, isNaturalEnd = false) {
  switch (action) {
    case "pause":
    case "resume":
    case "toggle-pause":
      if (!trainEventSource || completedEpochs < 1) return;
      if (!isPaused) {
        try {
          await fetch("/train/pause", { method: "POST" });
          isPaused = true;
          syncStudioUI("paused");
          appendConsoleLine(
            '<span class="term-yellow">// live training paused</span>\n',
          );
        } catch (e) {}
      } else {
        try {
          await fetch("/train/resume", { method: "POST" });
          isPaused = false;
          syncStudioUI("running");
          appendConsoleLine(
            '<span class="term-yellow">// live training resumed</span>\n',
          );
        } catch (e) {}
      }
      break;

    case "save":
      if (ui.btnSaveTrain.disabled || completedEpochs === 0) return;
      ui.btnSaveTrain.disabled = true;
      ui.btnSaveTrain.textContent = "saving...";
      try {
        const res = await fetch("/train/save", { method: "POST" });
        const data = await res.json();
        ui.btnSaveTrain.textContent = "saved!";
        if (data && data.stats) {
          updateStatsUI(data.stats);
          ui.telTime.textContent = `Time: ${data.stats.training_duration_sec.toFixed(2)}s`;
        }
        appendConsoleLine(
          `<span class="term-blue">[CHECKPOINT] Epoch ${completedEpochs} weights & stats saved to disk</span>\n`,
        );
      } catch (err) {
        ui.btnSaveTrain.textContent = "err";
      } finally {
        setTimeout(() => {
          if (isPaused) {
            ui.btnSaveTrain.textContent = "save";
            ui.btnSaveTrain.disabled = false;
          }
        }, 1500);
      }
      break;

    case "stop":
      if (!isNaturalEnd && (!trainEventSource || completedEpochs < 1)) return;
      if (!isNaturalEnd && trainEventSource) {
        try {
          const res = await fetch("/train/stop", { method: "POST" });
          const data = await res.json();
          if (data && data.stats && data.stats.epochs > 0) {
            updateStatsUI(data.stats);
            ui.telTime.textContent = `Time: ${data.stats.training_duration_sec.toFixed(2)}s`;
          }
        } catch (e) {}
      }
      const wasActive =
        !isNaturalEnd &&
        (ui.studioStatus.textContent === "training live..." ||
          ui.studioStatus.textContent === "paused");
      if (trainEventSource) {
        trainEventSource.close();
        trainEventSource = null;
      }
      isPaused = false;
      if (!isNaturalEnd) {
        syncStudioUI("stopped");
        if (wasActive) {
          appendConsoleLine(
            `<span class="term-red">// training stopped & epoch ${completedEpochs} checkpoint saved to disk</span>\n`,
          );
        }
      }
      break;
  }
}

const togglePauseLiveTraining = () => handleTrainAction("toggle-pause");
const saveLiveTrainingCheckpoint = () => handleTrainAction("save");
const stopLiveTraining = (isNaturalEnd = false) =>
  handleTrainAction("stop", isNaturalEnd);

// Event Listeners & Observers
window.addEventListener("resize", () => {
  if (isPanelOpen) renderGraphs();
});
if (window.ResizeObserver) {
  const ro = new window.ResizeObserver(() => {
    if (isPanelOpen) renderGraphs();
  });
  ro.observe(ui.lossCanvas.parentElement);
  ro.observe(ui.accCanvas.parentElement);
}
if (document.fonts) {
  document.fonts.ready.then(() => {
    if (isPanelOpen) renderGraphs();
  });
}

// Global scope export for HTML bindings
window.toggleRetrainPanel = toggleRetrainPanel;
window.toggleStatsCollapse = toggleStatsCollapse;
window.toggleTelemetryDetails = toggleTelemetryDetails;
window.startLiveTraining = startLiveTraining;
window.togglePauseLiveTraining = togglePauseLiveTraining;
window.saveLiveTrainingCheckpoint = saveLiveTrainingCheckpoint;
window.stopLiveTraining = stopLiveTraining;
window.clearConsole = clearConsole;
