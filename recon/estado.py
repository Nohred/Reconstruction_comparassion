"""Tabla de avance: capturas (real y sintetica), etapas hechas y Chamfer por tecnica.

    ./run.sh estado
"""
import json

from common.config import load_config, repo_path

TECNICAS = ["profundidad", "fotogrametria", "gaussian", "nerf"]


def main():
    common = load_config("common")
    lay = common["capture_layout"]
    caps = []
    for fuente, key in (("real", "real_root"), ("sintetica", "synthetic_root")):
        root = repo_path(common["paths"][key])
        if root.is_dir():
            caps += [(fuente, d.name) for d in sorted(root.iterdir()) if (d / lay["session_file"]).exists()]
    rroot = repo_path(common["paths"]["recon_root"])
    for fuente in ("real", "sintetica"):
        if (rroot / fuente).is_dir():
            caps += [(fuente, d.name) for d in sorted((rroot / fuente).iterdir())
                     if d.is_dir() and (fuente, d.name) not in caps]
    print(f"{'captura':28s} {'entrada':8s} " + " ".join(f"{t[:11]:>13s}" for t in TECNICAS) + "   evaluada")
    print("-" * 100)
    for fuente, name in caps:
        rd = rroot / fuente / name
        est = rd / "estado"
        m = {}
        mj = rd / "evaluacion" / "metricas.json"
        if mj.exists():
            m = json.loads(mj.read_text("utf-8"))["tecnicas"]
        cells = []
        for t in TECNICAS:
            if t in m and m[t]["sin_icp"].get("puntos"):
                perfil = (m[t].get("info") or {}).get("perfil", "")
                tag = "" if perfil in ("", "completo") else "*"
                cells.append(f"{m[t]['sin_icp']['chamfer_mm']:7.2f} mm{tag}")
            elif (est / f"{t}.hecho").exists():
                cells.append("hecha (" + (est / f"{t}.hecho").read_text().split()[0] + ")")
            elif (rd / t / "ns-train.log").exists() or (rd / t / "colmap.log").exists():
                cells.append("en curso?")
            else:
                cells.append("-")
        ent = "si" if (est / "entrada.hecho").exists() else "-"
        ev = (est / "evaluar.hecho").read_text().strip() if (est / "evaluar.hecho").exists() else "-"
        print(f"{fuente + '/' + name:28s} {ent:8s} " + " ".join(f"{c:>13s}" for c in cells) + f"   {ev}")
    print("\nCeldas con mm = Chamfer sin ICP (marco del tablero); * = perfil rapido (no comparable).\n"
          "Detalle: Resultados/resumen.csv")


if __name__ == "__main__":
    main()
