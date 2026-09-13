import io
import base64
import numpy as np
from PIL import Image

def process_canvas_image(img):
    gray = img.convert("L")
    bbox = gray.getbbox() # bounding box of drawn strokes
    if not bbox:
        return np.zeros((784, 1)), ""

    # crop and fit into 20x20 box preserving aspect ratio
    cropped = gray.crop(bbox)
    w, h = cropped.size
    ratio = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # paste onto 28x28 canvas centered
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    # center of mass shift to match mnist distribution
    arr = np.array(canvas, dtype=np.float64)
    total = np.sum(arr)
    if total > 0:
        y, x = np.indices((28, 28))
        shift_x = int(round(14.0 - np.sum(x * arr) / total))
        shift_y = int(round(14.0 - np.sum(y * arr) / total))
        arr = np.roll(arr, (shift_y, shift_x), axis=(0, 1))

    # 140x140 nearest-neighbor preview for the ui
    preview = Image.fromarray(arr.astype(np.uint8)).resize((140, 140), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return (arr / 255.0).reshape(784, 1), b64

def decode_base64_image(data_uri):
    return Image.open(io.BytesIO(base64.b64decode(data_uri.split(",")[-1])))
