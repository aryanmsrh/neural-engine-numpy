// -------------------------------------------------------------
// Interactive Drawing Canvas & Digit Presets
// -------------------------------------------------------------

const drawCanvas = document.getElementById("drawCanvas");
const dctx = drawCanvas.getContext("2d");
const chkLiveDraw = document.getElementById("chkLiveDraw");

let isDrawing = false;
let drawTimeout = null;

function clearCanvas() {
  dctx.fillStyle = "#000000";
  dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}
clearCanvas();

dctx.strokeStyle = "#FFFFFF";
dctx.lineWidth = 19;
dctx.lineCap = "round";
dctx.lineJoin = "round";

function getCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
}

function startDraw(e) {
  e.preventDefault();
  isDrawing = true;
  const { x, y } = getCoords(e);
  dctx.beginPath();
  dctx.moveTo(x, y);
}

function doDraw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const { x, y } = getCoords(e);
  dctx.lineTo(x, y);
  dctx.stroke();

  if (chkLiveDraw.checked) {
    clearTimeout(drawTimeout);
    drawTimeout = setTimeout(predictDrawnCanvas, 100);
  }
}

function stopDraw() {
  if (!isDrawing) return;
  isDrawing = false;
  if (chkLiveDraw.checked) predictDrawnCanvas();
}

drawCanvas.addEventListener("mousedown", startDraw);
drawCanvas.addEventListener("mousemove", doDraw);
window.addEventListener("mouseup", stopDraw);
drawCanvas.addEventListener("touchstart", startDraw, { passive: false });
drawCanvas.addEventListener("touchmove", doDraw, { passive: false });
window.addEventListener("touchend", stopDraw);

function drawPreset(digit) {
  clearCanvas();
  dctx.save();
  dctx.font = "bold 200px 'JetBrains Mono', monospace";
  dctx.fillStyle = "#FFFFFF";
  dctx.textAlign = "center";
  dctx.textBaseline = "middle";
  dctx.fillText(digit, drawCanvas.width / 2, drawCanvas.height / 2 + 10);
  dctx.restore();
  predictDrawnCanvas();
}

// Render digit 5 on load
window.addEventListener("DOMContentLoaded", () => drawPreset(5));