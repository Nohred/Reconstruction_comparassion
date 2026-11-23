"""Figuras y tablas del articulo (IEEE TIM) a partir de las evaluaciones.

    tools/ns_py.sh -m recon.articulo        (./run.sh articulo; compila Articulo/main.tex)

Lee Reconstrucciones/{real,sintetica}/<captura>/evaluacion/metricas.json, las capturas, los logs de
validacion del gemelo y Articulo/datos/mediciones_impresion.yaml (configs/recon.yaml articulo) y
escribe en Articulo/:
  figuras/gemelo.pdf      foto real vs render del gemelo (misma camara y parada), todas las capturas
  figuras/limpieza.pdf    que quita la limpieza en cada tecnica (una captura)
  figuras/error.pdf       nube limpia coloreada por distancia al GT, real vs gemelo (una captura)
  figuras/brecha.pdf      Chamfer real vs gemelo por tecnica y captura
  figuras/fscore.pdf      F-score vs umbral, media sobre capturas, real vs gemelo
  tablas/*.tex            capturas, impresion, resultados, resumen
Texto en ingles (la revista lo exige); los nombres cortos salen de la config.
"""
import json
import re

import numpy as np
import yaml

from common.config import load_config, repo_path
from recon.comun import MM, load_entrada, load_recon_config
from recon.informe import LIMPIEZA_COLOR, SERIE, TXT, TXT2, GRID, load_all, subsample
from recon.limpieza import CONSERVADO, FLOTANTE, OUTLIER, PISO, clean_cloud

COL1, COL2 = 3.5, 7.16                       # ancho de columna / pagina IEEE (in)
LIMPIEZA_EN = {CONSERVADO: "kept", PISO: "floor (board plane)", OUTLIER: "statistical outlier",
               FLOTANTE: "floating cluster"}
FUENTES = ("real", "sintetica")


def estilo(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
                         "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                         "ytick.color": TXT2, "text.color": TXT, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})


def metr(ev, fu, c):
    return ev.get((fu, c))


def val(m, t, k, bloque="sin_icp"):
    r = (m or {}).get("tecnicas", {}).get(t)
    return None if r is None else r.get(bloque, {}).get(k)


# ----------------------------------------------------------------------------- figuras
def fig_gemelo(cfg, caps, path, plt):
    import cv2
    y0, y1, x0, x1 = cfg["gemelo_recorte_px"]
    roots = {"real": repo_path(load_config("common")["paths"]["real_root"]),
             "sintetica": repo_path(load_config("common")["paths"]["synthetic_root"])}
    fig, axs = plt.subplots(2, len(caps), figsize=(COL2, 2.25), squeeze=False)
    for j, c in enumerate(caps):
        for i, fu in enumerate(FUENTES):
            f = roots[fu] / c / cfg["gemelo_camara"] / cfg["gemelo_parada"] / "color.png"
            im = cv2.imread(str(f))[y0:y1, x0:x1, ::-1]
            ax = axs[i][j]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cfg["nombres"][c], fontsize=7)
            if j == 0:
                ax.set_ylabel(("real" if fu == "real" else "digital twin"), fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.03, right=1, top=0.92, bottom=0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_limpieza(cfg, lcfg, ev, path, plt, o3d, rng):
    fu, c = cfg["limpieza_captura"]
    m = metr(ev, fu, c)
    rdir = m["_dir"]
    roi = load_entrada(rdir)["roi"]
    tecs = list(cfg["tecnicas"])
    fig, axs = plt.subplots(len(tecs), 1, figsize=(COL1, 0.8 * len(tecs) + 0.45), squeeze=False,
                            sharex=True, sharey=True, layout="constrained")
    htop = 0.0
    for i, t in enumerate(tecs):
        P = np.asarray(o3d.io.read_point_cloud(str(rdir / t / "resultado.ply")).points)
        _, info, (Q, lab) = clean_cloud(P, roi, lcfg, o3d, etiquetas=True)
        sel = subsample(len(Q), cfg["puntos_figura"], rng)
        Q, lab = Q[sel], lab[sel]
        ax = axs[i][0]
        for code in (PISO, CONSERVADO, OUTLIER, FLOTANTE):
            k = lab == code
            big = code in (OUTLIER, FLOTANTE)
            ax.scatter(Q[k, 0] / MM, -Q[k, 2] / MM, s=2.5 if big else 0.25, c=LIMPIEZA_COLOR[code][1], lw=0,
                       rasterized=True)
        htop = max(htop, float(np.percentile(-Q[lab == CONSERVADO, 2] / MM, 99.9)))
        p = info["pasos"]
        nr = max(p["roi"], 1)
        ax.set_ylabel(f"{cfg['tecnicas'][t]}\nz (mm)")
        ax.set_title(f"removed: floor {100 * (p['roi'] - p['sin_piso']) / nr:.1f}%, outliers "
                     f"{100 * (p['sin_piso'] - p['sin_outliers']) / nr:.1f}%, floaters "
                     f"{100 * (p['sin_outliers'] - p['final']) / nr:.1f}%  (cut {info['corte_mm']:.1f} mm)",
                     loc="left", color=TXT2, fontsize=6)
    axs[-1][0].set_xlabel("x (mm)")
    axs[0][0].set_ylim(-0.25 * htop, 1.6 * htop)     # flotantes mas altos quedan fuera (se cuentan en el titulo)
    h = [plt.Line2D([], [], marker="o", ls="", color=LIMPIEZA_COLOR[k][1], ms=3) for k in LIMPIEZA_EN]
    fig.legend(h, list(LIMPIEZA_EN.values()), ncol=4, loc="outside upper center", handletextpad=0.1,
               columnspacing=0.8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_error(cfg, ecfg, ev, path, plt, o3d, rng):
    import matplotlib

    from recon.evaluar import point_to_mesh_mm
    c = cfg["error_captura"]
    tecs = list(cfg["tecnicas"])
    vmax = ecfg["error_ply_max_mm"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "err", matplotlib.colormaps["Blues"](np.linspace(0.3, 1.0, 256)))
    fig, axs = plt.subplots(2, len(tecs), figsize=(COL2, 1.9), squeeze=False, sharey=True, layout="constrained")
    sc = None
    for i, fu in enumerate(FUENTES):
        m = metr(ev, fu, c)
        rdir = m["_dir"]
        mesh = o3d.io.read_triangle_mesh(str(rdir / "evaluacion" / "gt_tablero_m.ply"))
        G = np.asarray(mesh.sample_points_uniformly(cfg["puntos_figura"]).points)
        for j, t in enumerate(tecs):
            P = np.asarray(o3d.io.read_point_cloud(str(rdir / t / "resultado_limpio.ply")).points)
            P = P[subsample(len(P), cfg["puntos_figura"], rng)]
            e = point_to_mesh_mm(P, mesh, o3d)
            o = np.argsort(e)
            ax = axs[i][j]
            ax.scatter(G[:, 0] / MM, -G[:, 2] / MM, s=0.2, c=GRID, lw=0, rasterized=True)
            sc = ax.scatter(P[o, 0] / MM, -P[o, 2] / MM, s=0.4, c=e[o], cmap=cmap, vmin=0, vmax=vmax, lw=0,
                            rasterized=True)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)
            s = m["tecnicas"][t]["sin_icp"]
            ax.set_title(f"{cfg['tecnicas'][t]} {'real' if fu == 'real' else 'twin'}: "
                         f"CD {s['chamfer_mm']:.2f} mm", fontsize=6, loc="left")
            if j == 0:
                ax.set_ylabel("z (mm)", fontsize=6)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label(f"distance to GT (mm, clipped at {vmax:g})", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_brecha(cfg, ev, caps, path, plt):
    tecs = list(cfg["tecnicas"])
    fig, axs = plt.subplots(1, len(tecs), figsize=(COL2, 1.55), sharey=True, squeeze=False, layout="constrained")
    ys = np.arange(len(caps))[::-1]
    for j, t in enumerate(tecs):
        ax = axs[0][j]
        for y, c in zip(ys, caps):
            vr, vs = val(metr(ev, "real", c), t, "chamfer_mm"), val(metr(ev, "sintetica", c), t, "chamfer_mm")
            if vr is None or vs is None:
                continue
            ax.plot([vs, vr], [y, y], color=GRID, lw=1.6, zorder=1)
            ax.scatter([vs], [y], s=18, facecolor="white", edgecolor=SERIE[t], lw=1.3, zorder=3)
            ax.scatter([vr], [y], s=18, color=SERIE[t], zorder=3)
        ax.set_yticks(ys, [cfg["nombres"][c] for c in caps])
        ax.grid(axis="y", visible=False)
        ax.set_title(cfg["tecnicas"][t], loc="left")
        ax.set_xlabel("Chamfer distance (mm)")
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TXT2, mew=1.3, ms=4),
         plt.Line2D([], [], marker="o", ls="", color=TXT2, ms=4)]
    fig.legend(h, ["digital twin", "real"], ncol=2, loc="outside upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_fscore(cfg, ev, caps, path, plt):
    tecs = list(cfg["tecnicas"])
    fig, ax = plt.subplots(figsize=(COL1, 2.0), layout="constrained")
    for t in tecs:
        for fu, ls in zip(FUENTES, ("-", "--")):
            curvas, u = [], None
            for c in caps:
                r = (metr(ev, fu, c) or {}).get("tecnicas", {}).get(t, {}).get("sin_icp", {}).get("curva")
                if r:
                    u = np.array(r["umbral_mm"])
                    pr, rc = np.array(r["precision"]), np.array(r["recall"])
                    curvas.append(np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0.0))
            if curvas:
                ax.plot(u, np.mean(curvas, 0), ls=ls, color=SERIE[t], lw=1.2,
                        label=f"{cfg['tecnicas'][t]} {'real' if fu == 'real' else 'twin'}")
    ax.set_xlabel("threshold $\\tau$ (mm)")
    ax.set_ylabel("F-score (mean over objects)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- tablas
def validacion(cfg, c):
    n = (cfg.get("validacion_logs") or {}).get(c, c)
    f = repo_path("logs") / f"{n}_validacion.log"
    out = {}
    if f.exists():
        for line in f.read_text("utf-8", errors="replace").splitlines():
            m = re.search(r"IoU med ([0-9.]+)", line)
            if m and line.startswith("resumen"):
                out["iou"] = float(m.group(1))
            m = re.search(r"render med ([0-9.]+) px", line)
            if m and line.startswith("resumen"):
                out["render_px"] = float(m.group(1))
    return out


def tabla_capturas(cfg, caps, o3d):
    syn = repo_path(load_config("common")["paths"]["synthetic_root"])
    objs = yaml.safe_load(open(repo_path("configs/objects.yaml"), encoding="utf-8"))["reales"]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Object & Size (mm) & Material & Reproj. (px) & IoU \\", r"\midrule"]
    for c in caps:
        mesh = o3d.io.read_triangle_mesh(str(syn / c / "gt" / "objeto_rev.ply"))
        ext = mesh.get_axis_aligned_bounding_box().get_extent()
        v = validacion(cfg, c)
        clave = json.loads((repo_path(load_config("common")["paths"]["real_root"]) / c / "session.json")
                           .read_text("utf-8"))["objeto"]          # session.json[objeto] -> objects.yaml reales
        mat = cfg["materiales"].get(objs.get(clave, {}).get("material"), "--")
        L.append(f"{cfg['nombres'][c]} & {ext[0]:.0f}$\\times${ext[1]:.0f}$\\times${ext[2]:.0f} & {mat} & "
                 f"{v.get('render_px', float('nan')):.3f} & {v.get('iou', float('nan')):.3f} \\\\")
    return L + [r"\bottomrule", r"\end{tabular}"]


def tabla_impresion(cfg):
    d = yaml.safe_load(open(repo_path(cfg["mediciones"]), encoding="utf-8"))
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Feature & CAD (mm) & Printed (mm) & $\Delta$ (mm) \\",
         r"\midrule"]
    diffs = []
    for r in d["rasgos"]:
        dd = r["medido"] - r["nominal"]
        diffs.append(dd)
        L.append(f"{r['rasgo']} & {r['nominal']:.1f} & {r['medido']:.1f} & {dd:+.1f} \\\\")
    a = np.abs(diffs)
    L += [r"\midrule",
          f"\\multicolumn{{3}}{{l}}{{MAE / RMS / max $|\\Delta|$ (n = {len(a)})}} & "
          f"{a.mean():.2f} / {np.sqrt(np.mean(np.square(diffs))):.2f} / {a.max():.1f} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    return L, {"mae": float(a.mean()), "rms": float(np.sqrt(np.mean(np.square(diffs)))), "max": float(a.max()),
               "n": len(a)}


METRICAS = [("chamfer_mm", "CD (mm)", "menor", "{:.2f}", True), ("precision_mediana_mm", "Acc. (mm)", "menor",
            "{:.2f}", False), ("completitud_mediana_mm", "Comp. (mm)", "menor", "{:.2f}", False),
            ("fscore_1mm", "F@1", "mayor", "{:.2f}", False), ("fscore_2mm", "F@2", "mayor", "{:.2f}", True),
            ("fscore_5mm", "F@5", "mayor", "{:.2f}", False), ("hausdorff_p95_mm", "H95 (mm)", "menor", "{:.1f}",
                                                             False)]


def tabla_resultados(cfg, ev, caps):
    tecs = list(cfg["tecnicas"])
    spec = "ll" + "".join("rr" + ("r" if d else "") for *_, d in METRICAS)
    top, sub, rules, col = [], [], [], 3
    for k, nom, mej, f, d in METRICAS:
        n = 2 + d
        flecha = r"$\downarrow$" if mej == "menor" else r"$\uparrow$"
        top.append(f"\\multicolumn{{{n}}}{{c}}{{{nom} {flecha}}}")
        sub += ["R", "T"] + (["$\\Delta$"] if d else [])
        rules.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    L = [f"\\begin{{tabular}}{{{spec}}}", r"\toprule", "& & " + " & ".join(top) + r" \\", " ".join(rules),
         "Object & Method & " + " & ".join(sub) + r" \\", r"\midrule"]
    for ci, c in enumerate(caps):
        mr, ms = metr(ev, "real", c), metr(ev, "sintetica", c)
        best = {}
        for k, _, mej, _, _ in METRICAS:
            for fu, m in (("r", mr), ("s", ms)):
                vals = [val(m, t, k) for t in tecs]
                ok = [v for v in vals if v is not None]
                b = (min(ok) if mej == "menor" else max(ok)) if ok else None
                best[k, fu] = b
        for ti, t in enumerate(tecs):
            cells = []
            for k, _, _, f, d in METRICAS:
                vr, vs = val(mr, t, k), val(ms, t, k)
                for v, fu in ((vr, "r"), (vs, "s")):
                    s = "--" if v is None else f.format(v)
                    cells.append(f"\\textbf{{{s}}}" if v is not None and np.isclose(v, best[k, fu]) else s)
                if d:
                    cells.append("--" if None in (vr, vs) else ("{:+" + f[2:]).format(vr - vs))
            first = f"\\multirow{{{len(tecs)}}}{{*}}{{{cfg['nombres'][c]}}}" if ti == 0 else ""
            L.append(f"{first} & {cfg['tecnicas'][t]} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if ci < len(caps) - 1 else r"\bottomrule")
    return L + [r"\end{tabular}"]


def tiempos():
    """{(fuente, tecnica): [min]} de los logs de fondo (ultima corrida por captura)."""
    pat = re.compile(r"\] (real|sintetica)/(\S+): (profundidad|fotogrametria|gaussian|nerf) OK en (\d+) min (\d+) s")
    last = {}
    for f in sorted(repo_path("logs").glob("fondo_*.log")):
        for line in f.read_text("utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                last[m.group(1), m.group(2), m.group(3)] = int(m.group(4)) + int(m.group(5)) / 60
    out = {}
    for (fu, c, t), v in last.items():
        out.setdefault((fu, t), []).append(v)
    return out


def tabla_resumen(cfg, ev, caps):
    tecs = list(cfg["tecnicas"])
    tm = tiempos()

    def media(fu, t, k, bloque="sin_icp"):
        v = [val(metr(ev, fu, c), t, k, bloque) for c in caps]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{CD (mm) $\downarrow$} & \multicolumn{2}{c}{F@2 $\uparrow$} & "
         r"\multicolumn{2}{c}{Real + ICP} & Time \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
         r"Method & R & T & R/T & R & T & CD & shift & (min) \\", r"\midrule"]
    res = {}
    for t in tecs:
        cr, cs = media("real", t, "chamfer_mm"), media("sintetica", t, "chamfer_mm")
        fr, fs = media("real", t, "fscore_2mm"), media("sintetica", t, "fscore_2mm")
        ci = media("real", t, "chamfer_mm", "con_icp")
        sh = [metr(ev, "real", c)["tecnicas"][t].get("icp_desplazamiento_mm") for c in caps
              if t in metr(ev, "real", c)["tecnicas"]]
        sh = float(np.mean([x for x in sh if x is not None]))
        ti = tm.get(("real", t), []) + tm.get(("sintetica", t), [])
        res[t] = dict(cr=cr, cs=cs, fr=fr, fs=fs, ci=ci, sh=sh)
        L.append(f"{cfg['tecnicas'][t]} & {cr:.2f} & {cs:.2f} & {cr / cs:.1f} & {fr:.2f} & {fs:.2f} & {ci:.2f} & "
                 f"{sh:.1f} & {np.median(ti):.{0 if np.median(ti) >= 10 else 1}f} \\\\")
    return L + [r"\bottomrule", r"\end{tabular}"], res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    rc = load_recon_config()
    cfg = rc["articulo"]
    estilo(plt)
    rng = np.random.default_rng(cfg["semilla"])
    ev = {(m["fuente"], m["captura"]): m for m in load_all()}
    caps = [c for c in cfg["capturas"] if ("real", c) in ev and ("sintetica", c) in ev]
    out = repo_path(cfg["dir"])
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    (out / "tablas").mkdir(parents=True, exist_ok=True)

    fig_gemelo(cfg, caps, out / "figuras" / "gemelo.pdf", plt)
    fig_limpieza(cfg, rc["limpieza"], ev, out / "figuras" / "limpieza.pdf", plt, o3d, rng)
    fig_error(cfg, rc["evaluacion"], ev, out / "figuras" / "error.pdf", plt, o3d, rng)
    fig_brecha(cfg, ev, caps, out / "figuras" / "brecha.pdf", plt)
    fig_fscore(cfg, ev, caps, out / "figuras" / "fscore.pdf", plt)

    w = lambda n, L: (out / "tablas" / n).write_text("\n".join(L) + "\n", "utf-8")
    w("capturas.tex", tabla_capturas(cfg, caps, o3d))
    Li, imp = tabla_impresion(cfg)
    w("impresion.tex", Li)
    w("resultados.tex", tabla_resultados(cfg, ev, caps))
    Lr, res = tabla_resumen(cfg, ev, caps)
    w("resumen.tex", Lr)
    mac = {"printmae": f"{imp['mae']:.2f}", "printrms": f"{imp['rms']:.2f}", "printmax": f"{imp['max']:.1f}",
           "nobj": str(len(caps))}
    for t, r in res.items():
        k = {"nerf": "nerf", "gaussian": "gs", "fotogrametria": "mvs", "profundidad": "rgbd"}[t]
        mac.update({f"{k}R": f"{r['cr']:.2f}", f"{k}S": f"{r['cs']:.2f}", f"{k}D": f"{r['cr'] - r['cs']:+.2f}",
                    f"{k}FR": f"{r['fr']:.2f}", f"{k}FS": f"{r['fs']:.2f}", f"{k}ICP": f"{r['ci']:.2f}",
                    f"{k}shift": f"{r['sh']:.1f}"})
    w("numeros.tex", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()])
    (out / "datos" / "numeros.json").write_text(json.dumps({"capturas": caps, "impresion": imp, "resumen": res},
                                                           indent=1), "utf-8")
    print(f"-> {out}/figuras, {out}/tablas ({len(caps)} pares: {', '.join(caps)})")
    print(json.dumps({"impresion": imp, "resumen": res}, indent=1))


if __name__ == "__main__":
    main()
