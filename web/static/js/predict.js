// -------------------------------------------------------------
// Predict & Softmax Probability Display
// -------------------------------------------------------------

const topDigit = document.getElementById("topDigit");
const topConfidence = document.getElementById("topConfidence");
const timeDetail = document.getElementById("timeDetail");
const latencyBadgeSpan = document.querySelector("#latencyBadge span");
const modelVisionImg = document.getElementById("modelVisionImg");
const probContainer = document.getElementById("probBars");

// Generate 0-9 probability bars
let barsHtml = "";
for (let i = 0; i < 10; i++) {
  barsHtml += `
    <div class="prob-row" id="row-${i}">
      <div class="digit-idx">${i}</div>
      <div class="bar-track"><div class="bar-value" id="bar-${i}"></div></div>
      <div class="prob-val" id="pct-${i}">0.00%</div>
    </div>
  `;
}
probContainer.innerHTML = barsHtml;

const probRows = [];
for (let i = 0; i < 10; i++) {
  probRows.push({
    row: document.getElementById(`row-${i}`),
    bar: document.getElementById(`bar-${i}`),
    pct: document.getElementById(`pct-${i}`),
  });
}

function updateResults(data) {
  topDigit.textContent = data.prediction;
  topConfidence.textContent = (data.confidence * 100).toFixed(2) + "%";
  const tauMs = parseFloat(data.inference_time_ms).toFixed(4);
  timeDetail.innerHTML = `&tau;<sub>fwd</sub>: ${tauMs} ms`;
  if (latencyBadgeSpan) {
    latencyBadgeSpan.textContent = `fwd: ${parseFloat(data.inference_time_ms).toFixed(2)} ms`;
  }
  modelVisionImg.src = data.preview;

  for (let i = 0; i < 10; i++) {
    const pctStr = (data.probabilities[i] * 100).toFixed(2) + "%";
    const { row, bar, pct } = probRows[i];
    const isTop = i === data.prediction;

    bar.style.width = pctStr;
    pct.textContent = pctStr;
    row.classList.toggle("top-pick", isTop);
  }
}

async function predictDrawnCanvas() {
  const canvas = document.getElementById("drawCanvas");
  if (!canvas) return;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });
    updateResults(await res.json());
  } catch (e) {}
}

window.predictDrawnCanvas = predictDrawnCanvas;
window.updateResults = updateResults;
