"""Reporte corto en LaTeX: capturas reales vs su gemelo sintetico, por tecnica.

    ./run.sh reporte            (= tools/ns_py.sh -m recon.reporte)

Lee Reconstrucciones/{real,sintetica}/<captura>/evaluacion/metricas.json (pares que existen en ambas
fuentes, menos configs/recon.yaml reporte.excluir) y escribe Resultados/<reporte.dir>/reporte.tex;
si `reporte.compilar` (latexmk) esta instalado, tambien reporte.pdf.
  Tabla 1: detalle tecnica x captura, real / sintetico por metrica (+ diferencia), ICP de la real.
  Tabla 2: media por tecnica sobre las capturas y lugar (1 = mejor) por fuente.
Negritas = mejor tecnica de esa captura, fuente y metrica.
"""
import shutil
import subprocess

import numpy as np

from common.config import load_config, repo_path
from recon.comun import load_recon_config
from recon.informe import load_all, pares

FUENTES = ("real", "sint")


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def valor(m, t, k):
    r = m["tecnicas"].get(t)
    return None if r is None else r.get("sin_icp", {}).get(k)


def celda(v, f, negrita=False):
    if v is None:
        return "--"
    s = f.format(v)
    return rf"\textbf{{{s}}}" if negrita else s


def mejores(vals, mejor):
    """Indices de la mejor tecnica (con empates) en una lista con posibles None."""
    ok = [v for v in vals if v is not None]
    if not ok:
        return set()
    b = min(ok) if mejor == "menor" else max(ok)
    return {i for i, v in enumerate(vals) if v is not None and np.isclose(v, b)}


def encabezado(mets):
    ncol = sum(2 + bool(m["delta"]) for m in mets)
    spec = "ll" + "".join("rr" + ("r" if m["delta"] else "") for m in mets) + "rr"
    top, sub, rules, c = [], [], [], 3
    for m in mets:
        n = 2 + bool(m["delta"])
        u = f" ({m['unidad']})" if m["unidad"] else ""
        flecha = r"$\downarrow$" if m["mejor"] == "menor" else r"$\uparrow$"
        top.append(rf"\multicolumn{{{n}}}{{c}}{{\shortstack{{{m['nombre']}\\{u.strip()} {flecha}}}}}")
        sub += ["R", "S"] + ([r"$\Delta$"] if m["delta"] else [])
        rules.append(rf"\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    top.append(r"\multicolumn{2}{c}{\shortstack{Real\\con ICP}}")
    rules.append(rf"\cmidrule(lr){{{c}-{c + 1}}}")
    sub += ["Ch.", "mueve"]
    return spec, ncol + 2, [
        r"\toprule",
        "& & " + " & ".join(top) + r" \\",
        " ".join(rules),
        r"T\'ecnica & Captura & " + " & ".join(sub) + r" \\",
        r"\midrule",
    ]


def tabla_detalle(prs, tecs, mets):
    spec, ncol, lines = encabezado(mets)
    # mejor tecnica por captura, fuente y metrica
    best = {}
    for c, mr, ms in prs:
        for fu, m in zip(FUENTES, (mr, ms)):
            for me in mets:
                best[c, fu, me["clave"]] = mejores([valor(m, t, me["clave"]) for t in tecs], me["mejor"])
    for j, t in enumerate(tecs):
        acc = {}
        for i, (c, mr, ms) in enumerate(prs):
            cells = []
            for me in mets:
                k, f = me["clave"], me["fmt"]
                vr, vs = valor(mr, t, k), valor(ms, t, k)
                acc.setdefault(k, []).append((vr, vs))
                cells += [celda(vr, f, j in best[c, "real", k]), celda(vs, f, j in best[c, "sint", k])]
                if me["delta"]:
                    cells.append("--" if None in (vr, vs) else ("{:+" + f[2:]).format(vr - vs))
            rr = mr["tecnicas"].get(t, {})
            ci = rr.get("con_icp", {}).get("chamfer_mm")
            cells += [celda(ci, "{:.2f}"), celda(rr.get("icp_desplazamiento_mm"), "{:.1f}")]
            acc.setdefault("_icp", []).append((ci, rr.get("icp_desplazamiento_mm")))
            first = rf"\multirow{{{len(prs) + 1}}}{{*}}{{{tex(t)}}}" if i == 0 else ""
            lines.append(f"{first} & {tex(c)} & " + " & ".join(cells) + r" \\")
        cells = []
        for me in mets:
            k, f = me["clave"], me["fmt"]
            vr = [a for a, _ in acc[k] if a is not None]
            vs = [b for _, b in acc[k] if b is not None]
            mr_, ms_ = (float(np.mean(vr)) if vr else None), (float(np.mean(vs)) if vs else None)
            cells += [celda(mr_, f), celda(ms_, f)]
            if me["delta"]:
                cells.append("--" if None in (mr_, ms_) else ("{:+" + f[2:]).format(mr_ - ms_))
        ci = [a for a, _ in acc["_icp"] if a is not None]
        mv = [b for _, b in acc["_icp"] if b is not None]
        cells += [celda(np.mean(ci) if ci else None, "{:.2f}"), celda(np.mean(mv) if mv else None, "{:.1f}")]
        lines.append(r"\rowcolor{gris} & \textit{media} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if j < len(tecs) - 1 else r"\bottomrule")
    return [rf"\begin{{tabular}}{{{spec}}}"] + lines + [r"\end{tabular}"]


def medias(prs, tecs, mets):
    """{(tecnica, fuente, clave): media sobre capturas}"""
    out = {}
    for t in tecs:
        for me in mets:
            k = me["clave"]
            for fu, idx in zip(FUENTES, (1, 2)):
                v = [valor(p[idx], t, k) for p in prs]
                v = [x for x in v if x is not None]
                out[t, fu, k] = float(np.mean(v)) if v else None
    return out


def tabla_resumen(prs, tecs, mets):
    med = medias(prs, tecs, mets)
    spec = "l" + "rrr" * len(mets)
    top = " & ".join(rf"\multicolumn{{3}}{{c}}{{{m['nombre']}}}" for m in mets)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(mets)))
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", f"& {top} " + r"\\", rules,
             r"T\'ecnica & " + " & ".join(["R", "S", "lugar R/S"] * len(mets)) + r" \\", r"\midrule"]
    lugar = {}
    for me in mets:
        for fu in FUENTES:
            vals = [med[t, fu, me["clave"]] for t in tecs]
            orden = sorted((v, i) for i, v in enumerate(vals) if v is not None)
            if me["mejor"] == "mayor":
                orden = orden[::-1]
            for r, (_, i) in enumerate(orden, 1):
                lugar[tecs[i], fu, me["clave"]] = r
    for t in tecs:
        cells = []
        for me in mets:
            k, f = me["clave"], me["fmt"]
            lr, ls = lugar.get((t, "real", k)), lugar.get((t, "sint", k))
            cells += [celda(med[t, "real", k], f, lr == 1), celda(med[t, "sint", k], f, ls == 1),
                      f"{lr or '--'}/{ls or '--'}"]
        lines.append(f"{tex(t)} & " + " & ".join(cells) + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}"], med


def observaciones(prs, tecs, med, prs_icp):
    ch, f2 = "chamfer_mm", "fscore_2mm"
    obs = []
    br = min(tecs, key=lambda t: med[t, "real", ch])
    bs = min(tecs, key=lambda t: med[t, "sint", ch])
    obs.append(rf"Mejor t\'ecnica por Chamfer medio: \textbf{{{tex(br)}}} en real "
               rf"({med[br, 'real', ch]:.2f}\,mm) y \textbf{{{tex(bs)}}} en sint\'etico ({med[bs, 'sint', ch]:.2f}\,mm).")
    gaps = ", ".join(rf"{tex(t)} {med[t, 'real', ch] - med[t, 'sint', ch]:+.2f}\,mm" for t in tecs)
    obs.append(rf"Brecha real $-$ sint\'etico del Chamfer medio: {gaps}.")
    gf = ", ".join(rf"{tex(t)} {med[t, 'real', f2] - med[t, 'sint', f2]:+.2f}" for t in tecs)
    obs.append(rf"Brecha real $-$ sint\'etico del F@2mm medio: {gf}.")
    for t, mv, ci in prs_icp:
        obs.append(rf"En real, \textit{{{tex(t)}}} queda desplazada en promedio {mv:.1f}\,mm del marco del "
                   rf"tablero (ChArUco/RGB); alineada con ICP su Chamfer medio baja a {ci:.2f}\,mm.")
    return obs


GLOSARIO = [
    (r"Precisi\'on", r"distancia de cada punto reconstruido a la superficie GT (mediana). Mide ruido, "
     r"flotantes y desplazamientos."),
    ("Completitud", r"distancia de muestras de la superficie GT al punto reconstruido m\'as cercano "
     r"(mediana). Mide huecos y zonas no reconstruidas."),
    ("Chamfer", r"promedio de precisi\'on media y completitud media: error global en mm."),
    ("Hausdorff p95", r"percentil 95 de las distancias: error de los peores puntos (bordes, colas, flotantes)."),
    ("F@$t$", r"media arm\'onica de la fracci\'on de puntos a menos de $t$\,mm del GT y de la fracci\'on del "
     r"GT cubierta a menos de $t$\,mm (0--1). F@1 = detalle fino, F@2 = global, F@5 = forma gruesa."),
    ("ICP", r"Sin ICP (todas las columnas R/S) se mide en el marco de la captura: forma + posici\'on. "
     r"``Real con ICP'': Chamfer tras alinear la nube al GT (solo forma) y cu\'anto la movi\'o el ICP (mm)."),
]


def documento(cfg, prs, tecs, mets, excluidas):
    t1 = tabla_detalle(prs, tecs, mets)
    t2, med = tabla_resumen(prs, tecs, mets)
    icp = []
    for t in tecs:
        mv = [p[1]["tecnicas"][t].get("icp_desplazamiento_mm") for p in prs if t in p[1]["tecnicas"]]
        ci = [p[1]["tecnicas"][t].get("con_icp", {}).get("chamfer_mm") for p in prs if t in p[1]["tecnicas"]]
        mv, ci = [x for x in mv if x is not None], [x for x in ci if x is not None]
        if mv and ci and np.mean(mv) > cfg["icp_destacar_mm"]:
            icp.append((t, float(np.mean(mv)), float(np.mean(ci))))
    obs = observaciones(prs, tecs, med, icp)
    caps = ", ".join(tex(c) for c, _, _ in prs)
    exc = "".join(rf" Se excluye \textit{{{tex(c)}}}: {tex(r)}." for c, r in excluidas.items())
    L = [r"\documentclass[10pt]{article}",
         r"\usepackage[a4paper,landscape,margin=1.4cm]{geometry}",
         r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[spanish]{babel}",
         r"\usepackage{lmodern,booktabs,multirow,array}\usepackage[table]{xcolor}\usepackage{caption}",
         r"\definecolor{gris}{gray}{0.92}\captionsetup{font=small,labelfont=bf}",
         r"\setlength{\tabcolsep}{3pt}\pagestyle{empty}",
         r"\begin{document}",
         rf"\begin{{center}}{{\Large {cfg['titulo']}}}\\[2pt]{{\small \today}}\end{{center}}\vspace{{-2mm}}",
         r"\noindent{\small Cada captura real (3 L515 fijas, base giratoria con tablero ChArUco, 16 paradas) se "
         r"reprodujo en Isaac Sim con las mismas poses de c\'amara y de la base (gemelo sint\'etico). Ambas se "
         r"reconstruyeron con las mismas cuatro t\'ecnicas y la misma entrada (poses e intr\'insecos conocidos), "
         r"se limpiaron igual (ROI, piso, outliers, flotantes) y se compararon con la malla GT; en la real, el "
         rf"GT se registr\'o con ICP de 4 GDL (pose de consenso). Capturas: {caps}.{exc}}}",
         r"\par\vspace{2mm}\begin{center}\scriptsize",
         r"\captionof{table}{Detalle por t\'ecnica y captura. R = real, S = sint\'etico (gemelo), "
         r"$\Delta$ = R $-$ S. \textbf{Negrita}: mejor t\'ecnica en esa captura, fuente y m\'etrica. "
         r"$\downarrow$ menor es mejor, $\uparrow$ mayor es mejor.}"]
    L += t1
    L += [r"\end{center}",
          r"\begin{center}\scriptsize",
          r"\captionof{table}{Media por t\'ecnica sobre las capturas; lugar R/S = posici\'on entre las t\'ecnicas "
          r"(1 = mejor) en real y en sint\'etico.}"]
    L += t2
    L += [r"\end{center}", r"\par\noindent\begin{minipage}[t]{0.49\textwidth}\footnotesize",
          r"\textbf{M\'etricas} (contra la malla GT; la franja de la base no observable se excluye)",
          r"\begin{itemize}\setlength\itemsep{0pt}"]
    L += [rf"\item \textbf{{{a}}}: {b}" for a, b in GLOSARIO]
    L += [r"\end{itemize}\end{minipage}\hfill\begin{minipage}[t]{0.49\textwidth}\footnotesize",
          r"\textbf{Observaciones}", r"\begin{itemize}\setlength\itemsep{0pt}"]
    L += [rf"\item {o}" for o in obs]
    L += [r"\end{itemize}\end{minipage}", r"\end{document}"]
    return "\n".join(L) + "\n"


def main():
    cfg = load_recon_config()["reporte"]
    excl = cfg.get("excluir") or {}
    prs = [p for p in pares(load_all()) if p[0] not in excl]
    if not prs:
        raise SystemExit("no hay pares real/sintetico evaluados")
    out = repo_path(load_config("common")["paths"]["results_root"]) / cfg["dir"]
    out.mkdir(parents=True, exist_ok=True)
    texf = out / "reporte.tex"
    texf.write_text(documento(cfg, prs, cfg["tecnicas"], cfg["metricas"],
                              {c: r for c, r in excl.items() if c in {m["captura"] for m in load_all()}}), "utf-8")
    print(f"-> {texf} ({len(prs)} pares)")
    cmd = cfg["compilar"]
    if shutil.which(cmd[0]) is None:
        print(f"{cmd[0]} no instalado: solo .tex")
        return
    r = subprocess.run(cmd + [texf.name], cwd=out, capture_output=True, text=True, errors="replace")
    if r.returncode:
        print(r.stdout[-3000:])
        raise SystemExit("fallo la compilacion LaTeX")
    subprocess.run([cmd[0], "-c"], cwd=out, capture_output=True)
    print(f"-> {out / 'reporte.pdf'}")


if __name__ == "__main__":
    main()
