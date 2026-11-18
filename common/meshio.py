"""PLY binario de triangulos sin dependencias (malla GT del gemelo)."""
from pathlib import Path

import numpy as np


def write_mesh_ply(path, points, counts, indices):
    """PLY binario de triangulos (sin dependencias). Solo mallas trianguladas."""
    if not np.all(counts == 3):
        raise NotImplementedError("solo triangulos")
    P = np.asarray(points, np.float32)
    F = np.asarray(indices, np.int32).reshape(-1, 3)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    face = np.empty(len(F), dtype=[("n", "u1"), ("v", "<i4", (3,))])
    face["n"], face["v"] = 3, F
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(P.astype("<f4").tobytes())
        f.write(face.tobytes())


def read_mesh_ply(path):
    """Lee el PLY escrito por write_mesh_ply: (puntos (N,3) float32, caras (M,3) int32)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode("ascii").splitlines()
    nv = int(next(h for h in head if h.startswith("element vertex")).split()[-1])
    nf = int(next(h for h in head if h.startswith("element face")).split()[-1])
    P = np.frombuffer(data, "<f4", nv * 3, end).reshape(nv, 3)
    face = np.frombuffer(data, [("n", "u1"), ("v", "<i4", (3,))], nf, end + nv * 12)
    return P.astype(np.float64), face["v"].astype(np.int32)


def write_points_ply(path, points, colors=None):
    """Nube de puntos PLY binaria (x, y, z float32 y, opcional, red/green/blue uchar)."""
    P = np.asarray(points, "<f4").reshape(-1, 3)
    props = "property float x\nproperty float y\nproperty float z\n"
    dt = [("p", "<f4", (3,))]
    if colors is not None:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        dt.append(("c", "u1", (3,)))
    rec = np.empty(len(P), dtype=dt)
    rec["p"] = P
    if colors is not None:
        rec["c"] = np.asarray(colors, np.uint8).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n{props}end_header\n".encode())
        f.write(rec.tobytes())
