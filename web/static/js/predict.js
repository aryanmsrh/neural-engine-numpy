// -------------------------------------------------------------
// Predict & Softmax Probability Display
// -------------------------------------------------------------

const topDigit = document.getElementById("topDigit");
const topConfidence = document.getElementById("topConfidence");
const timeDetail = document.getElementById("timeDetail");
const latencyBadge = document.getElementById("latencyBadge");
const modelVisionImg = document.getElementById("modelVisionImg");
const probContainer = document.getElementById("probBars");

// Generate 0-9 probability bars
let barsHtml = "";
for (let i = 0; i < 10; i++) {
  barsHtml += `
    <div class="prob-row">
      <div class="digit-idx">${i}</div>
      <div class="bar-track"><div class="bar-value" id="bar-${i}"></div></div>
      <div class="prob-val" id="pct-${i}">0.00%</div>
    </div>
  `;
}
probContainer.innerHTML = barsHtml;

function updateResults(data) {
  topDigit.textContent = data.prediction;
  topConfidence.textContent = (data.confidence * 100).toFixed(2) + "%";
  const tauMs = parseFloat(data.inference_time_ms).toFixed(4);
  timeDetail.innerHTML = `&tau;<sub>fwd</sub>: ${tauMs} ms`;
  if (latencyBadge) {
    const span = latencyBadge.querySelector("span:last-child");
    if (span) span.innerHTML = `&tau;<sub>fwd</sub>: ${tauMs} ms`;
  }
  modelVisionImg.src = data.preview;

  for (let i = 0; i < 10; i++) {
    const pct = (data.probabilities[i] * 100).toFixed(2) + "%";
    const bar = document.getElementById(`bar-${i}`);
    const pctEl = document.getElementById(`pct-${i}`);
    bar.style.width = pct;
    pctEl.textContent = pct;

    const isTop = i === data.prediction;
    bar.classList.toggle("top-pick", isTop);
    pctEl.style.color = isTop ? "#ffffff" : "var(--dim)";
    pctEl.style.fontWeight = isTop ? "900" : "400";
  }
}

async function predictDrawnCanvas() {
  const canvas = document.getElementById("drawCanvas");
  const dataUri = canvas.toDataURL("image/png");
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUri }),
  });
  const data = await res.json();
  updateResults(data);
}
