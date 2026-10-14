const studioWrapper = document.getElementById("studioWrapper");
const studioStatus = document.getElementById("studioStatus");
const btnToggleRetrain = document.getElementById("btnToggleRetrain");
const btnStartTrain = document.getElementById("btnStartTrain");
const btnStopTrain = document.getElementById("btnStopTrain");

const inpEpochs = document.getElementById("inpEpochs");
const inpLR = document.getElementById("inpLR");
const inpBatch = document.getElementById("inpBatch");
const inpVerbose = document.getElementById("inpVerbose");

const trainProgFill = document.getElementById("trainProgFill");
const telEpoch = document.getElementById("telEpoch");
const telBatch = document.getElementById("telBatch");
const telLoss = document.getElementById("telLoss");
const telBatchAcc = document.getElementById("telBatchAcc");
const telTestAcc = document.getElementById("telTestAcc");
const telSpeed = document.getElementById("telSpeed");
const telTime = document.getElementById("telTime");
const telSamples = document.getElementById("telSamples");
const trainConsole = document.getElementById("trainConsole");

const lossCanvas = document.getElementById("lossCanvas");
const accCanvas = document.getElementById("accCanvas");
const lossHeadVal = document.getElementById("lossHeadVal");
const accHeadVal = document.getElementById("accHeadVal");

const statAccuracy = document.getElementById("statAccuracy");
const statLoss = document.getElementById("statLoss");
const statEpochs = document.getElementById("statEpochs");

let trainEventSource = null;
let isPanelOpen = false;
let lossHistory = [];
let accHistory = [];
let testHistory = [];
let startTime = 0;
let totalSteps = 0;

function toggleRetrainPanel() {
  isPanelOpen = !isPanelOpen;
  studioWrapper.style.display = isPanelOpen ? "block" : "none";
  btnToggleRetrain.classList.toggle("active", isPanelOpen);
  if (isPanelOpen) {
    requestAnimationFrame(() => {
      renderGraphs();
      studioWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  } else if (trainEventSource) {
    stopLiveTraining();
  }
}

function clearConsole() {
  if (trainConsole) trainConsole.innerHTML = "";
}

function pad(val, len) {
  return String(val).padStart(len, " ");
}

function fmtSec(sec) {
  return sec.toFixed(2).padStart(6, " ") + "S";
}

function appendConsoleLine(html) {
  if (!trainConsole) return;
  trainConsole.insertAdjacentHTML("beforeend", html);
  trainConsole.scrollTop = trainConsole.scrollHeight;
}

// Crisp HiDPI Canvas setup with 2x supersampling floor to eliminate any blurriness
function prepareCanvas(canvas) {
  const dpr = Math.max(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const parent = canvas.parentElement;
  const w = rect.width
    ? Math.round(rect.width)
    : parent && parent.clientWidth
      ? parent.clientWidth
      : 450;
  const h = rect.height
    ? Math.round(rect.height)
    : parent && parent.clientHeight
      ? parent.clientHeight
      : 210;

  const targetWidth = Math.round(w * dpr);
  const targetHeight = Math.round(h * dpr);

  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext("2d");
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return { ctx, w, h };
}

function renderLossGraph() {
  if (!lossCanvas) return;
  const { ctx, w, h } = prepareCanvas(lossCanvas);

  ctx.fillStyle = "#030303";
  ctx.fillRect(0, 0, w, h);

  const padLeft = 58;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 24;
  const pw = w - padLeft - padRight;
  const ph = h - padTop - padBottom;

  const N = lossHistory.length;
  let floorL = 0.0;
  let ceilL = 2.5;

  if (N > 0) {
    const minLoss = Math.min(...lossHistory);
    const maxLoss = Math.max(...lossHistory);
    const lossSpan = Math.max(maxLoss - minLoss, 0.15);
    floorL = Math.max(0, minLoss - lossSpan * 0.08);
    ceilL = maxLoss + lossSpan * 0.08;
  }

  // Gridlines & Loss values (rounded to 4 decimal points)
  ctx.font = "600 11px 'JetBrains Mono', monospace";
  [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
    const y = Math.round(padTop + (1 - f) * ph) + 0.5;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + pw, y);
    ctx.stroke();

    const val = floorL + f * (ceilL - floorL);
    ctx.fillStyle = "#f59e0b";
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(4), padLeft - 7, y + 4);
  });

  // Step Axis
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#666";
  ctx.textAlign = "left";
  ctx.fillText("Step 1", padLeft, h - 7);
  ctx.textAlign = "right";
  ctx.fillText(`Step ${totalSteps || N || 1}`, padLeft + pw, h - 7);

  if (N === 0) return;

  const xSpan = Math.max(totalSteps ? totalSteps - 1 : N - 1, 1);

  if (N === 1) {
    const x = padLeft;
    const y = padTop + (1 - (lossHistory[0] - floorL) / (ceilL - floorL)) * ph;
    ctx.fillStyle = "rgba(245, 158, 11, 0.35)";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
    if (lossHeadVal) lossHeadVal.textContent = lossHistory[0].toFixed(4);
    return;
  }

  // Gradient fill under loss
  const gradLoss = ctx.createLinearGradient(0, padTop, 0, padTop + ph);
  gradLoss.addColorStop(0, "rgba(245, 158, 11, 0.25)");
  gradLoss.addColorStop(1, "rgba(245, 158, 11, 0.0)");

  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = padLeft + (i / xSpan) * pw;
    const y = padTop + (1 - (lossHistory[i] - floorL) / (ceilL - floorL)) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 2.0;
  ctx.stroke();

  ctx.lineTo(padLeft + ((N - 1) / xSpan) * pw, padTop + ph);
  ctx.lineTo(padLeft, padTop + ph);
  ctx.closePath();
  ctx.fillStyle = gradLoss;
  ctx.fill();

  // Active glowing tip
  const lastX = padLeft + ((N - 1) / xSpan) * pw;
  const lastY =
    padTop + (1 - (lossHistory[N - 1] - floorL) / (ceilL - floorL)) * ph;
  ctx.fillStyle = "rgba(245, 158, 11, 0.35)";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#f59e0b";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, 2 * Math.PI);
  ctx.fill();

  if (lossHeadVal && N > 0) {
    lossHeadVal.textContent = lossHistory[N - 1].toFixed(4);
  }
}

function renderAccGraph() {
  if (!accCanvas) return;
  const { ctx, w, h } = prepareCanvas(accCanvas);

  ctx.fillStyle = "#030303";
  ctx.fillRect(0, 0, w, h);

  const padLeft = 58;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 24;
  const pw = w - padLeft - padRight;
  const ph = h - padTop - padBottom;

  const N = accHistory.length;
  let floorA = 0;
  let ceilA = 100;

  if (N > 0) {
    let minAcc = Math.min(...accHistory);
    let maxAcc = Math.max(...accHistory);

    // Dynamic range scaling: accuracy bounds adapt to training progress
    floorA = Math.max(0, Math.floor((minAcc - 4) / 5) * 5);
    ceilA = Math.min(100, Math.ceil((maxAcc + 4) / 5) * 5);
    if (ceilA - floorA < 20) {
      ceilA = Math.min(100, floorA + 20);
      floorA = Math.max(0, ceilA - 20);
    }
  }

  const spanA = Math.max(ceilA - floorA, 10);

  // Gridlines & Accuracy percentage values (rounded to 2 decimal points)
  ctx.font = "600 11px 'JetBrains Mono', monospace";
  [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {
    const y = Math.round(padTop + (1 - f) * ph) + 0.5;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + pw, y);
    ctx.stroke();

    const val = floorA + f * spanA;
    ctx.fillStyle = "#22c55e";
    ctx.textAlign = "right";
    ctx.fillText(`${val.toFixed(2)}%`, padLeft - 7, y + 4);
  });

  // Step Axis
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#666";
  ctx.textAlign = "left";
  ctx.fillText("Step 1", padLeft, h - 7);
  ctx.textAlign = "right";
  ctx.fillText(`Step ${totalSteps || N || 1}`, padLeft + pw, h - 7);

  if (N === 0) return;

  const xSpan = Math.max(totalSteps ? totalSteps - 1 : N - 1, 1);

  if (N === 1) {
    const x = padLeft;
    const y = padTop + (1 - (accHistory[0] - floorA) / spanA) * ph;
    ctx.fillStyle = "rgba(34, 197, 94, 0.35)";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
    if (accHeadVal)
      accHeadVal.textContent = `${accHistory[0].toFixed(2)}%`;
    return;
  }

  // Gradient fill under Batch Accuracy
  const gradAcc = ctx.createLinearGradient(0, padTop, 0, padTop + ph);
  gradAcc.addColorStop(0, "rgba(34, 197, 94, 0.22)");
  gradAcc.addColorStop(1, "rgba(34, 197, 94, 0.0)");

  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = padLeft + (i / xSpan) * pw;
    const y = padTop + (1 - (accHistory[i] - floorA) / spanA) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2.0;
  ctx.stroke();

  ctx.lineTo(padLeft + ((N - 1) / xSpan) * pw, padTop + ph);
  ctx.lineTo(padLeft, padTop + ph);
  ctx.closePath();
  ctx.fillStyle = gradAcc;
  ctx.fill();

  // Active glowing tip for batch accuracy
  const lastX = padLeft + ((N - 1) / xSpan) * pw;
  const lastY = padTop + (1 - (accHistory[N - 1] - floorA) / spanA) * ph;
  ctx.fillStyle = "rgba(34, 197, 94, 0.35)";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#22c55e";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, 2 * Math.PI);
  ctx.fill();

  if (accHeadVal && N > 0) {
    accHeadVal.textContent = accHistory[N - 1].toFixed(2) + "%";
  }
}

function renderGraphs() {
  renderLossGraph();
  renderAccGraph();
}

function startLiveTraining() {
  const epochs = parseInt(inpEpochs.value) || 10;
  const lr = parseFloat(inpLR.value) || 0.1;
  const batchSize = parseInt(inpBatch.value) || 64;
  const verbose = parseInt(inpVerbose ? inpVerbose.value : 2);

  btnStartTrain.disabled = true;
  btnStopTrain.disabled = false;
  studioStatus.className = "phase-badge phase-forward";
  studioStatus.textContent = "training live...";

  if (statAccuracy) statAccuracy.textContent = "--%";
  if (statLoss) statLoss.textContent = "--";
  if (statEpochs) statEpochs.textContent = "0";
  if (lossHeadVal) lossHeadVal.textContent = "--";
  if (accHeadVal) accHeadVal.textContent = "--%";

  lossHistory = [];
  accHistory = [];
  testHistory = [];
  renderGraphs();
  clearConsole();

  let stepCounter = 0;
  let totalEpochs = epochs;
  let totalBatches = 0;
  totalSteps = 0;
  startTime = 0;

  trainEventSource = new EventSource(
    `/train?epochs=${epochs}&lr=${lr}&batch_size=${batchSize}`,
  );

  trainEventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === "start" || data.type === "init") {
      startTime = data.time;
      totalEpochs = data.total_epochs;
      totalBatches = data.total_batches;
      totalSteps = totalEpochs * totalBatches;

      telEpoch.textContent = `Epoch: 1/${totalEpochs}`;
      telBatch.textContent = `Batch: 1/${totalBatches}`;
      if (telSamples)
        telSamples.textContent = `${data.train_samples.toLocaleString()} train / ${data.test_samples.toLocaleString()} test`;

      if (verbose >= 1) {
        appendConsoleLine(
          `<span class="term-red">[  0.00S] STRT // SMPLS: ${pad(data.train_samples, 6)} // EPOC: ${pad(totalEpochs, 6)} // BTCH: ${pad(batchSize, 6)}</span>\n\n`,
        );
      }
      if (verbose >= 2) {
        appendConsoleLine(
          `<span class="term-yellow">    [  0.00S] EPOC ${pad(0, 6)}/${pad(totalEpochs, 6)} | LOSS: ${pad("-", 6)} | TACC: ${pad("-", 6)}%</span>\n`,
        );
      }
    } else if (data.type === "step" || data.type === "batch_complete") {
      stepCounter = data.step;
      lossHistory.push(data.loss);
      accHistory.push(data.batch_accuracy);

      const pct = Math.min(100, (stepCounter / totalSteps) * 100);
      trainProgFill.style.width = pct + "%";
      telBatch.textContent = `Batch: ${data.batch}/${totalBatches}`;
      telLoss.textContent = `Loss: ${data.loss.toFixed(4)}`;
      telBatchAcc.textContent = `Batch Acc: ${data.batch_accuracy.toFixed(2)}%`;

      const elapsed = Math.max(0.01, data.time - startTime);
      if (telTime) telTime.textContent = `Time: ${elapsed.toFixed(1)}s`;
      if (telSpeed) {
        const speed = Math.round((stepCounter * batchSize) / elapsed);
        telSpeed.textContent = `Speed: ${speed.toLocaleString()} smp/s`;
      }

      if (verbose >= 3) {
        appendConsoleLine(
          `<span class="term-dim">       [${fmtSec(elapsed)}] STEP ${pad(data.step, 6)} (BTCH ${pad(data.batch, 6)}) | LOSS: ${data.loss.toFixed(4).padStart(6, " ")} | BACC: ${data.batch_accuracy.toFixed(2).padStart(6, " ")}%</span>\n`,
        );
      }

      renderGraphs();
    } else if (data.type === "epoch" || data.type === "epoch_complete") {
      telTestAcc.textContent = `Test Acc: ${data.test_accuracy.toFixed(2)}%`;
      if (data.epoch < totalEpochs)
        telEpoch.textContent = `Epoch: ${data.epoch + 1}/${totalEpochs}`;

      testHistory.push({
        step: stepCounter || data.epoch * totalBatches,
        acc: data.test_accuracy,
        avg_loss: data.avg_loss,
        epoch: data.epoch,
      });

      const elapsed = data.time - startTime;
      if (verbose >= 2) {
        appendConsoleLine(
          `<span class="term-yellow">    [${fmtSec(elapsed)}] EPOC ${pad(data.epoch, 6)}/${pad(totalEpochs, 6)} | LOSS: ${data.avg_loss.toFixed(4).padStart(6, " ")} | TACC: ${data.test_accuracy.toFixed(2).padStart(6, " ")}%</span>\n`,
        );
      }

      renderGraphs();
    } else if (data.type === "end" || data.type === "finished") {
      stopLiveTraining();
      studioStatus.className = "phase-badge phase-forward";
      studioStatus.textContent = "completed";
      trainProgFill.style.width = "100%";

      statAccuracy.textContent = data.final_accuracy.toFixed(2) + "%";
      statLoss.textContent = data.final_loss.toFixed(4);
      statEpochs.textContent = totalEpochs;

      const elapsed = data.time - startTime;
      if (verbose >= 1) {
        appendConsoleLine(
          `\n<span class="term-red">[${fmtSec(elapsed)}] OVER // FACC: ${data.final_accuracy.toFixed(2).padStart(6, " ")}% // FLOS: ${data.final_loss.toFixed(4).padStart(6, " ")} // DURN: ${fmtSec(elapsed)}</span>\n`,
        );
      }
    }
  };

  trainEventSource.onerror = () => stopLiveTraining();
}

function stopLiveTraining() {
  if (trainEventSource) {
    trainEventSource.close();
    trainEventSource = null;
  }
  btnStartTrain.disabled = false;
  btnStopTrain.disabled = true;
  if (studioStatus.textContent === "training live...") {
    studioStatus.textContent = "stopped";
    studioStatus.className = "phase-badge phase-idle";
  }
}

window.addEventListener("resize", () => {
  if (isPanelOpen) renderGraphs();
});

if (window.ResizeObserver) {
  const ro = new window.ResizeObserver(() => {
    if (isPanelOpen) renderGraphs();
  });
  if (lossCanvas && lossCanvas.parentElement)
    ro.observe(lossCanvas.parentElement);
  if (accCanvas && accCanvas.parentElement) ro.observe(accCanvas.parentElement);
}

if (document.fonts) {
  document.fonts.ready.then(() => {
    if (isPanelOpen) renderGraphs();
  });
}
