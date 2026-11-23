"""Informe comparativo de todas las evaluaciones: reconstruccion (nube limpia) vs ground truth.

    ./run.sh informe            (= tools/ns_py.sh -m recon.informe)

Lee Reconstrucciones/*/*/evaluacion/metricas.json (lo escribe recon.evaluar) y genera
Resultados/evaluacion/ (configs/recon.yaml informe):
  informe.md                       metodo, tablas resumen y por captura, enlaces a figuras
  figuras/resumen_chamfer.png      Chamfer por captura y tecnica (sin ICP)
  figuras/resumen_fscore.png       F-score vs umbral, una grafica por captura
  figuras/real_vs_sintetica.png    cada captura real frente a su gemelo sintetico (si existen ambas)
  figuras/<fuente>_<captura>_limpieza.png   que quito la limpieza (piso / outliers / flotantes)
  figuras/<fuente>_<captura>_error.png      nube limpia coloreada por distancia al GT (2 vistas)
"""
import json
import time

import numpy as np

from common.config import load_config, repo_path
from recon.comun import MM, load_entrada, load_recon_config
from recon.limpieza import CONSERVADO, FLOTANTE, OUTLIER, PISO, clean_cloud

TECNICAS = ["profundidad", "fotogrametria", "gaussian", "nerf"]
# paleta categorica de referencia (orden fijo, slots 1-4) y tinta de texto
SERIE = {"profundidad": "#2a78d6", "fotogrametria": "#eb6834", "gaussian": "#1baf7a", "nerf": "#eda100"}
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LIMPIEZA_COLOR = {CONSERVADO: ("conservado", "#2a78d6"), PISO: ("piso (plano del tablero)", "#c9c8c2"),
                  OUTLIER: ("outlier estadistico", "#eb6834"), FLOTANTE: ("cluster flotante", "#e34948")}


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
                         "text.color": TXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "legend.frameon": False})


def load_all():
    common = load_config("common")
    out = []
    for mj in sorted(repo_path(common["paths"]["recon_root"]).glob("*/*/evaluacion/metricas.json")):
        m = json.loads(mj.read_text("utf-8"))
        m["_dir"] = mj.parent.parent
        out.append(m)
    return out


def perfil(r):
    return (r.get("info") or {}).get("perfil", "")


def label(m):
    return f"{m['fuente']}/{m['captura']}"


def fig_chamfer(evals, path, plt):
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(evals) + 2), 3.6), layout="constrained")
    w = 0.8 / len(TECNICAS)
    for j, t in enumerate(TECNICAS):
        xs, ys = [], []
        for i, m in enumerate(evals):
            r = m["tecnicas"].get(t, {}).get("sin_icp", {})
            if r.get("puntos"):
                xs.append(i + (j - 1.5) * w)
                ys.append(r["chamfer_mm"])
        ax.bar(xs, ys, w * 0.9, color=SERIE[t], label=t, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7, color=TXT2)
    ax.set_xticks(range(len(evals)))
    ax.set_xticklabels([label(m) + ("\n(perfil rapido)" if any(perfil(r) == "rapido" for r in m["tecnicas"].values())
                                    else "") for m in evals])
    ax.set_ylabel("Chamfer (mm), menor = mejor")
    ax.grid(axis="x", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="outside upper left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_fscore(evals, path, plt):
    n = len(evals)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows + 0.4), squeeze=False, sharey=True,
                            layout="constrained")
    for k, m in enumerate(evals):
        ax = axs[k // cols][k % cols]
        for t in TECNICAS:
            c = m["tecnicas"].get(t, {}).get("sin_icp", {}).get("curva")
            if not c:
                continue
            p, r = np.array(c["precision"]), np.array(c["recall"])
            f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
            ax.plot(c["umbral_mm"], f, color=SERIE[t], lw=2, label=t)
        ax.set_title(label(m), fontsize=9, loc="left")
        ax.set_xlabel("umbral (mm)")
        ax.set_ylim(0, 1)
        if k % cols == 0:
            ax.set_ylabel("F-score")
    for k in range(n, rows * cols):
        axs[k // cols][k % cols].axis("off")
    fig.legend(*axs[0][0].get_legend_handles_labels(), ncol=4, loc="outside upper left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def pares(evals):
    """[(captura, real, sintetica)] de las capturas evaluadas en ambas fuentes (gemelos)."""
    by = {(m["fuente"], m["captura"]): m for m in evals}
    return [(c, by[("real", c)], by[("sintetica", c)]) for f, c in sorted(by) if f == "real" and ("sintetica", c) in by]


PAR_METRICAS = [("chamfer_mm", "Chamfer (mm), menor = mejor"), ("fscore_2mm", "F@2mm, mayor = mejor")]


def fig_pares(prs, path, plt):
    """Dumbbell por tecnica: sintetica (hueco) -> real (lleno) para cada captura."""
    fig, axs = plt.subplots(len(PAR_METRICAS), len(TECNICAS), figsize=(11, 0.45 * len(prs) * 2 + 2.4),
                            sharey=True, sharex="row", squeeze=False, layout="constrained")
    ys = np.arange(len(prs))[::-1]
    for i, (key, xl) in enumerate(PAR_METRICAS):
        for j, t in enumerate(TECNICAS):
            ax = axs[i][j]
            for y, (c, mr, ms) in zip(ys, prs):
                vr = mr["tecnicas"].get(t, {}).get("sin_icp", {}).get(key)
                vs = ms["tecnicas"].get(t, {}).get("sin_icp", {}).get(key)
                if vr is None or vs is None:
                    continue
                ax.plot([vs, vr], [y, y], color=GRID, lw=2, zorder=1)
                ax.scatter([vs], [y], s=46, facecolor=SURF, edgecolor=SERIE[t], lw=2, zorder=3,
                           label="sintetica (gemelo)" if y == ys[0] else None)
                ax.scatter([vr], [y], s=46, color=SERIE[t], edgecolor=SURF, lw=1.5, zorder=3,
                           label="real" if y == ys[0] else None)
            ax.set_yticks(ys)
            ax.set_yticklabels([c for c, _, _ in prs])
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(xl, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=9, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", mfc=SURF, mec=TXT2, mew=2, ms=7),
         plt.Line2D([], [], marker="o", ls="", color=TXT2, ms=7)]
    fig.legend(h, ["sintetica (gemelo)", "real"], ncol=2, loc="outside upper left",
               title="real vs su gemelo sintetico, por tecnica (sin ICP)", title_fontsize=10, alignment="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tabla_pares(prs, key, f):
    lines = ["| captura | " + " | ".join(f"{t} real / sint" for t in TECNICAS) + " |",
             "|---|" + "---:|" * len(TECNICAS)]
    gaps = {t: [] for t in TECNICAS}
    for c, mr, ms in prs:
        cells = []
        for t in TECNICAS:
            rr, rs = mr["tecnicas"].get(t, {}), ms["tecnicas"].get(t, {})
            vr, vs = rr.get("sin_icp", {}).get(key), rs.get("sin_icp", {}).get(key)
            star = "*" if "rapido" in (perfil(rr), perfil(rs)) else ""
            if vr is not None and vs is not None:
                gaps[t].append(vr - vs)
            cells.append(f"{fmt(vr, f)} / {fmt(vs, f)}{star}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("| **media real - sint** | " + " | ".join(
        fmt(float(np.mean(g)) if g else None, "{:+" + f[2:]) for g in gaps.values()) + " |")
    return lines


def views(P):
    """Dos vistas ortograficas (marco tablero, m -> mm): frente (x, altura) y lado (y, altura)."""
    return [(P[:, 0] / MM, -P[:, 2] / MM, "x (mm)"), (P[:, 1] / MM, -P[:, 2] / MM, "y (mm)")]


def subsample(n, k, rng):
    return np.arange(n) if n <= k else rng.choice(n, k, replace=False)


def fig_limpieza(m, icfg, lcfg, o3d, path, plt, rng):
    rdir = m["_dir"]
    roi = load_entrada(rdir)["roi"]
    tecs = [t for t in TECNICAS if (rdir / t / "resultado.ply").exists()]
    fig, axs = plt.subplots(len(tecs), 2, figsize=(8, 2.4 * len(tecs) + 0.6), squeeze=False, layout="constrained")
    for i, t in enumerate(tecs):
        P = np.asarray(o3d.io.read_point_cloud(str(rdir / t / "resultado.ply")).points)
        _, info, (Q, lab) = clean_cloud(P, roi, lcfg, o3d, etiquetas=True)
        sel = subsample(len(Q), icfg["puntos_figura"], rng)
        Q, lab = Q[sel], lab[sel]
        for j, (x, y, xl) in enumerate(views(Q)):
            ax = axs[i][j]
            for code in (PISO, CONSERVADO, OUTLIER, FLOTANTE):     # lo quitado encima, visible
                k = lab == code
                big = code in (OUTLIER, FLOTANTE)
                ax.scatter(x[k], y[k], s=4 if big else 0.4, c=LIMPIEZA_COLOR[code][1], lw=0, rasterized=True,
                           label=LIMPIEZA_COLOR[code][0] if i == j == 0 else None)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{t}\naltura (mm)", fontsize=8)
                p = info["pasos"]
                nr = max(p["roi"], 1)
                ax.set_title(f"quitado: piso {100 * (p['roi'] - p['sin_piso']) / nr:.1f}%, outliers "
                             f"{100 * (p['sin_piso'] - p['sin_outliers']) / nr:.1f}%, flotantes "
                             f"{100 * (p['sin_outliers'] - p['final']) / nr:.1f}%", fontsize=7.5, loc="left",
                             color=TXT2)
    fig.legend(*axs[0][0].get_legend_handles_labels(), ncol=4, loc="outside upper left", markerscale=5,
               title=f"{label(m)}: limpieza de la nube dentro de la ROI", title_fontsize=10, alignment="left")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_error(m, icfg, ecfg, o3d, path, plt, rng):
    import matplotlib

    from recon.evaluar import point_to_mesh_mm

    rdir = m["_dir"]
    mesh = o3d.io.read_triangle_mesh(str(rdir / "evaluacion" / "gt_tablero_m.ply"))
    tecs = [t for t in TECNICAS if (rdir / t / "resultado_limpio.ply").exists()]
    vmax = ecfg["error_ply_max_mm"]
    G = np.asarray(mesh.sample_points_uniformly(icfg["puntos_figura"]).points)
    fig, axs = plt.subplots(len(tecs), 2, figsize=(8.6, 2.4 * len(tecs) + 0.5), squeeze=False, layout="constrained")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "err", matplotlib.colormaps["Blues"](np.linspace(0.3, 1.0, 256)))
    sc = None
    for i, t in enumerate(tecs):
        P = np.asarray(o3d.io.read_point_cloud(str(rdir / t / "resultado_limpio.ply")).points)
        P = P[subsample(len(P), icfg["puntos_figura"], rng)]
        e = point_to_mesh_mm(P, mesh, o3d)
        o = np.argsort(e)                                    # errores grandes encima
        for j, ((x, y, xl), (gx, gy, _)) in enumerate(zip(views(P[o]), views(G))):
            ax = axs[i][j]
            ax.scatter(gx, gy, s=0.3, c=GRID, lw=0, rasterized=True)
            sc = ax.scatter(x, y, s=0.6, c=e[o], cmap=cmap, vmin=0, vmax=vmax, lw=0, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=8)
            if j == 0:
                s = m["tecnicas"][t]["sin_icp"]
                ax.set_ylabel(f"{t}\naltura (mm)", fontsize=8)
                ax.set_title(f"precision mediana {s.get('precision_mediana_mm', float('nan')):.2f} mm, "
                             f"p90 {s.get('precision_p90_mm', float('nan')):.2f} mm", fontsize=7.5, loc="left",
                             color=TXT2)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, shrink=0.6, pad=0.02)
        cb.set_label(f"distancia a la malla GT (mm, saturado en {vmax:g})")
    fig.suptitle(f"{label(m)}: error de la nube limpia (gris claro = GT)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def tabla_resumen(evals, key, f):
    lines = ["| captura | " + " | ".join(TECNICAS) + " |", "|---|" + "---:|" * len(TECNICAS)]
    for m in evals:
        cells = []
        for t in TECNICAS:
            r = m["tecnicas"].get(t, {})
            v = r.get("sin_icp", {}).get(key)
            cells.append(fmt(v, f) + ("*" if v is not None and perfil(r) == "rapido" else ""))
        lines.append(f"| {label(m)} | " + " | ".join(cells) + " |")
    return lines


def tabla_captura(m, umbrales):
    head = (["tecnica", "pts ROI", "corte piso", "piso %", "outl. %", "flot. %", "pts limpios", "precision med", "precision p90",
             "complet. med", "complet. p90", "chamfer", "hausdorff p95"]
            + [f"F@{u:g}mm" for u in umbrales] + ["chamfer c/ICP", "ICP mueve"])
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for t in TECNICAS:
        r = m["tecnicas"].get(t)
        if not r:
            continue
        s, c, p = r["sin_icp"], r.get("con_icp", {}), (r.get("limpieza") or {}).get("pasos", {})
        roi = p.get("roi") or 0
        pct = (lambda a, b: f"{100 * (p[a] - p[b]) / roi:.1f}" if roi and a in p else "-")
        lines.append("| " + " | ".join(
            [t + ("*" if perfil(r) == "rapido" else ""), str(roi or "-"),
             fmt((r.get("limpieza") or {}).get("corte_mm"), "{:.1f}"), pct("roi", "sin_piso"),
             pct("sin_piso", "sin_outliers"), pct("sin_outliers", "final"), str(s.get("puntos", 0)),
             fmt(s.get("precision_mediana_mm")), fmt(s.get("precision_p90_mm")),
             fmt(s.get("completitud_mediana_mm")), fmt(s.get("completitud_p90_mm")),
             fmt(s.get("chamfer_mm")), fmt(s.get("hausdorff_p95_mm"))]
            + [fmt(s.get(f"fscore_{u:g}mm"), "{:.3f}") for u in umbrales]
            + [fmt(c.get("chamfer_mm")), fmt(r.get("icp_desplazamiento_mm"), "{:.1f}")]) + " |")
    return lines


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    style(plt)
    common, rcfg = load_config("common"), load_recon_config()
    icfg, ecfg, lcfg = rcfg["informe"], rcfg["evaluacion"], rcfg["limpieza"]
    out = repo_path(common["paths"]["results_root"]) / icfg["dir"]
    figs = out / "figuras"
    figs.mkdir(parents=True, exist_ok=True)
    evals = load_all()
    if not evals:
        raise SystemExit("no hay evaluaciones (./run.sh evaluar <captura>)")
    rng = np.random.default_rng(icfg["semilla"])
    fig_chamfer(evals, figs / "resumen_chamfer.png", plt)
    fig_fscore(evals, figs / "resumen_fscore.png", plt)
    um = ecfg["umbrales_fscore_mm"]
    pl = lcfg["plano"]
    md = [f"# Evaluacion: reconstruccion vs ground truth",
          "", f"Generado {time.strftime('%Y-%m-%d %H:%M')} con `./run.sh informe` a partir de "
          "`Reconstrucciones/*/*/evaluacion/metricas.json`. No editar a mano.", "",
          "## Metodo", "",
          "Todas las nubes estan en el marco del tablero ChArUco (mm). Antes de comparar, cada nube "
          "(`<tecnica>/resultado.ply`) se limpia igual para las 4 tecnicas (`recon/limpieza.py`, "
          "`configs/recon.yaml: limpieza`) y se guarda en `<tecnica>/resultado_limpio.ply`:", "",
          f"1. **ROI**: cilindro de radio {common['recon']['roi']['radio_mm']:g} mm sobre el eje de la base.",
          f"2. **Piso**: plano del tablero ajustado con RANSAC (umbral {pl['ransac_dist_mm']:g} mm) en la propia "
          f"nube, con los puntos con |z| < {pl['busqueda_mm']:g} mm de la corona exterior de la ROI "
          f"(r > {pl['anillo_frac']:g} R, donde nunca hay objeto). Si el ajuste es dudoso (inclinacion > "
          f"{pl['max_inclinacion_deg']:g} deg o a mas de {pl['max_desplazamiento_mm']:g} mm de z = 0) se usa z = 0. "
          f"En la corona se mide el grosor del piso reconstruido (percentil {pl['grosor_percentil']:g} de la "
          f"altura sobre el plano) y se quita todo lo que queda por debajo de **corte = max("
          f"{lcfg['sobre_plano_mm']:g} mm, grosor + {pl['margen_mm']:g} mm)** (tope {lcfg['corte_max_mm']:g} mm).",
          f"3. **Outliers estadisticos**: distancia media a {lcfg['estadistico']['vecinos']} vecinos > media + "
          f"{lcfg['estadistico']['std']:g} sigma.",
          f"4. **Clusters flotantes**: DBSCAN (eps {lcfg['clusters']['eps_mm']:g} mm sobre la nube a "
          f"{lcfg['clusters']['voxel_mm']:g} mm); semilla = el mayor cluster que toca el piso (a < "
          f"{lcfg['clusters']['contacto_mm']:g} mm del corte); se suman los que tengan >= "
          f"{100 * lcfg['clusters']['min_fraccion']:g}% de los puntos y esten a < {lcfg['clusters']['union_mm']:g} mm; "
          "el resto son flotantes.", "",
          "El GT es la malla del objeto colocada en la escena sintetica, muestreada uniformemente; la franja "
          "junto a su base (apoyada en el tablero, no observable) se excluye hasta la **misma altura del "
          "corte del piso de cada tecnica**, para no penalizar la completitud por el corte. En capturas reales la pose "
          "del objeto no se conoce: el GT se registra (barrido de yaw + ICP de 4 GDL: x, y, z, yaw, sin inclinar) "
          "sobre la nube limpia de CADA tecnica y se usa la pose con menor error mediano en el conjunto (consenso); "
          "esa pose sirve para todas las tecnicas. El marco del tablero lo fijan las camaras de color (ChArUco), asi "
          "que el consenso lo deciden las tecnicas RGB; un sesgo de la profundidad de las L515 aparece como error de "
          "profundidad *sin ICP* (y desaparece *con ICP*).", "",
          "Metricas (mm): **precision** = distancia de cada punto reconstruido a la malla GT; **completitud** "
          "= distancia de cada punto del GT (muestreado) a la nube; **chamfer** = media de ambas medias; "
          "**F@t** = media armonica de la fraccion de puntos con precision < t y de GT con completitud < t. "
          "*Sin ICP* = en las poses de la captura (lo que importa); *con ICP* = tras alinear la nube al GT "
          "(separa error de forma de error de registro).", "",
          "`*` = perfil rapido (pocas iteraciones; solo valida el flujo, no comparable).", "",
          "## Resumen", "", "Chamfer sin ICP (mm, menor = mejor):", "", *tabla_resumen(evals, "chamfer_mm", "{:.2f}"),
          "", "F-score a 2 mm sin ICP (mayor = mejor):", "", *tabla_resumen(evals, "fscore_2mm", "{:.3f}"), "",
          "![Chamfer](figuras/resumen_chamfer.png)", "", "![F-score](figuras/resumen_fscore.png)", ""]
    prs = pares(evals)
    if prs:
        fig_pares(prs, figs / "real_vs_sintetica.png", plt)
        md += ["## Real vs sintetico (gemelo)", "",
               "Cada captura real se compara con su gemelo (`./run.sh gemelo`: mismas poses de camara y de la "
               "base, mismo objeto, escala y material). Las dos se evaluan contra la MISMA malla GT; en la real va "
               "registrada por consenso (ver Metodo). La diferencia real - sintetico mide lo que el gemelo "
               "no reproduce (ruido y sesgo de las L515, iluminacion, material, calibracion).", "",
               "Chamfer sin ICP (mm), real / sintetico:", "", *tabla_pares(prs, "chamfer_mm", "{:.2f}"), "",
               "F@2mm sin ICP, real / sintetico:", "", *tabla_pares(prs, "fscore_2mm", "{:.3f}"), "",
               "![real vs sintetico](figuras/real_vs_sintetica.png)", ""]
    for m in evals:
        tag = f"{m['fuente']}_{m['captura']}"
        print(f"figuras de {label(m)} ...")
        fig_limpieza(m, icfg, lcfg, o3d, figs / f"{tag}_limpieza.png", plt, rng)
        fig_error(m, icfg, ecfg, o3d, figs / f"{tag}_error.png", plt, rng)
        md += [f"## {label(m)}", "", f"GT: `{m['gt']}`" + (
            f"; registrado por consenso, pose de {m['registro_gt'].get('referencia', 'profundidad')} (fitness "
            f"{m['registro_gt']['fitness']:.2f}, rmse {m['registro_gt']['rmse_mm']:.2f} mm; error mediano por pose "
            "candidata: " + ", ".join(f"{k} {v:.2f}" for k, v in m['registro_gt'].get('candidatas_mm', {}).items())
            + " mm)" if "registro_gt" in m else ""), "",
            *tabla_captura(m, um), "", "Plano del tablero ajustado por tecnica:", ""]
        for t in TECNICAS:
            p = (m["tecnicas"].get(t, {}).get("limpieza") or {}).get("plano")
            if p:
                md.append(f"- {t}: corte {p['corte_mm']:.1f} mm (grosor del piso {p['grosor_mm']:.1f} mm), "
                          f"plano {p['metodo']}" + (
                    f", inclinacion {p['inclinacion_deg']:.2f} deg, z=0 a {p['desplazamiento_mm']:+.2f} mm "
                    f"del plano, rms {p['rms_mm']:.2f} mm" if p["metodo"] == "ransac" else f" ({p['motivo']})"))
        md += ["", f"![limpieza](figuras/{tag}_limpieza.png)", "", f"![error](figuras/{tag}_error.png)", "",
               f"Nubes para MeshLab/CloudCompare: `{m['_dir'].relative_to(repo_path('.'))}/evaluacion/"
               "<tecnica>_error.ply` (color = error) y `<tecnica>/resultado_limpio.ply`.", ""]
    (out / "informe.md").write_text("\n".join(md), "utf-8")
    print(f"-> {out / 'informe.md'} ({len(evals)} capturas)")


if __name__ == "__main__":
    main()
