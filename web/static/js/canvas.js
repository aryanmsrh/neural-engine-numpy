// -------------------------------------------------------------
// Interactive Drawing Canvas & Digit Presets
// -------------------------------------------------------------

const drawCanvas = document.getElementById("drawCanvas");
const dctx = drawCanvas.getContext("2d");
const chkLiveDraw = document.getElementById("chkLiveDraw");

let isDrawing = false;
let drawTimeout = null;

function clearCanvas() {
  dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
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
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function startDraw(e) {
  isDrawing = true;
  const { x, y } = getCoords(e);
  dctx.beginPath();
  dctx.moveTo(x, y);
}

function doDraw(e) {
  if (!isDrawing) return;
  const { x, y } = getCoords(e);
  dctx.lineTo(x, y);
  dctx.stroke();

  if (chkLiveDraw && chkLiveDraw.checked) {
    clearTimeout(drawTimeout);
    drawTimeout = setTimeout(predictDrawnCanvas, 100);
  }
}

function stopDraw() {
  if (!isDrawing) return;
  isDrawing = false;
  if (chkLiveDraw && chkLiveDraw.checked) predictDrawnCanvas();
}

drawCanvas.addEventListener("pointerdown", startDraw);
drawCanvas.addEventListener("pointermove", doDraw);
window.addEventListener("pointerup", stopDraw);
window.addEventListener("pointercancel", stopDraw);

function drawPreset(digit) {
  clearCanvas();
  dctx.save();
  dctx.font = "bold 200px Arial, Helvetica, sans-serif";
  dctx.fillStyle = "#FFFFFF";
  dctx.textAlign = "center";
  dctx.textBaseline = "middle";
  dctx.fillText(digit, drawCanvas.width / 2, drawCanvas.height / 2 + 10);
  dctx.restore();
  predictDrawnCanvas();
}

window.clearCanvas = clearCanvas;
window.drawPreset = drawPreset;

// Render digit 5 on load
window.addEventListener("DOMContentLoaded", () => drawPreset(5));