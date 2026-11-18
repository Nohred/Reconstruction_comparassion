"""Genera una vista previa reducida (JPG, <= 640 px de ancho) para poder VER una imagen.

    tools/inspect/run.sh make_preview.py Captura_real/bunnny/alta/stop_00/color.png
    tools/inspect/run.sh make_preview.py real.png sintetica.png --name comp_alta_00   # lado a lado

Varias entradas se ponen lado a lado (cada una escalada a la misma altura) y el conjunto se
reduce a --max-width. La profundidad uint16 se colorea (percentiles 1-99, invalidos en negro).
Salida en tmp/previews/. Solo se lee el JPG resultante, nunca el original.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "tmp" / "previews"
MAX_W = 640
JPG_QUALITY = 85
LABEL_H = 22


def to_bgr(img):
    if img.dtype == np.uint16 and img.ndim == 2:          # profundidad
        valid = img > 0
        if valid.any():
            lo, hi = np.percentile(img[valid], (1, 99))
            norm = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        else:
            norm = np.zeros(img.shape, np.float32)
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[~valid] = 0
        return col
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", help="nombre del JPG de salida (sin extension)")
    ap.add_argument("--max-width", type=int, default=MAX_W)
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()
    width = min(args.max_width, MAX_W)

    tiles = []
    for f in args.files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"No se pudo leer {f}")
        tiles.append((Path(f), to_bgr(img), img.shape))
    h = min(t[1].shape[0] for t in tiles)
    row = []
    for p, img, _ in tiles:
        s = h / img.shape[0]
        row.append(cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA))
    canvas = np.hstack(row)
    s = width / canvas.shape[1]
    if s < 1:
        canvas = cv2.resize(canvas, (width, int(round(canvas.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if not args.no_labels:
        x = 0
        for (p, _, _), r in zip(tiles, row):
            label = "/".join(p.parts[-3:])
            cv2.putText(canvas, label, (x + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(canvas, label, (x + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            x += int(round(r.shape[1] * min(s, 1)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or "__".join(p.parent.name + "_" + p.stem for p, _, _ in tiles)
    out = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    print(f"{out}  ({canvas.shape[1]}x{canvas.shape[0]}; originales "
          f"{', '.join(f'{sh[1]}x{sh[0]}' for _, _, sh in tiles)})")


if __name__ == "__main__":
    main()
