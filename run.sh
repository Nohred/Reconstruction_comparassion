#!/usr/bin/env bash
# Orquestador: captura sintetica (Isaac Sim) -> entrada comun -> 4 reconstrucciones -> evaluacion.
#   ./run.sh ayuda
set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$ROOT"
ISAAC="${ISAACSIM_PATH:-$HOME/isaacsim}"
FORCE="${FORCE:-0}"
export PERFIL="${PERFIL:-completo}"
TECNICAS_TODAS="profundidad fotogrametria gaussian nerf"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

ayuda() {
    cat <<'EOF'
Uso: ./run.sh <etapa> [argumentos]          (FORCE=1 rehace, PERFIL=rapido para pruebas)

Captura sintetica (Isaac Sim) -> Captura_sintetica/<nombre>/   (mismo formato que la real)
  gemelo  Captura_real/<captura>            gemelo exacto de una captura real (sus poses)
  modelo  <ruta.usd[z]|clave> [opciones]    cualquier objeto con la plantilla + variabilidad
          --nombre X  --semilla N  --tamano-mm 170 | --escala F | --tamano-real
  lote    <carpeta con .usd/.usdz> [opciones de modelo]   un 'modelo' por archivo (+ reconstruir)
  objetos [claves de configs/objects.yaml...]   modelo + reconstruir + informe por objeto
                                            (sin claves = todos los de objects.yaml)
  pares   [capturas de Captura_real/...]    gemelo + reconstruir real y sintetica + informe
                                            (sin capturas = todas las de Captura_real/)

Reconstruccion: <captura> = real/<nombre> | sintetica/<nombre>  -> Reconstrucciones/<captura>/
  entrada       <captura>   imagenes sin distorsion, mascaras, COLMAP (poses fijas + puntos),
                            transforms.json (nerfstudio), vistas.json            [segundos]
  profundidad   <captura>   nubes L515 + ICP multivista + TSDF                   [~1 min]
  fotogrametria <captura>   COLMAP MVS (patch-match + fusion)                    [~10-30 min]
  gaussian      <captura>   splatfacto (gsplat) + profundidad renderizada + TSDF [~20-40 min]
  nerf          <captura>   nerfacto + profundidad renderizada + TSDF            [~30-60 min]
  limpiar       <captura>   solo limpieza de las nubes (ROI, piso, outliers, flotantes)
                            -> <tecnica>/resultado_limpio.ply                     [segundos]
  evaluar       <captura>   limpieza + metricas vs malla GT -> evaluacion/, Resultados/resumen.csv
  reconstruir   <captura> [tecnicas...]   entrada + tecnicas (todas por defecto) + evaluar + informe
  informe                   Resultados/evaluacion/informe.md + figuras (todas las capturas evaluadas)
  reporte                   Resultados/reporte/reporte.tex (+ .pdf): real vs sintetico, tabla detallada
  articulo                  Articulo/: figuras + tablas (recon.articulo) y main.pdf (IEEEtran, latexmk)

Utilidades
  fondo <etapa> [args]      corre la etapa en segundo plano (sobrevive al cerrar la terminal);
                            log en logs/; seguir con: tail -f logs/<archivo>
  estado                    tabla de avance y metricas de todas las capturas
  pruebas                   pruebas unitarias

Ejemplos
  ./run.sh gemelo Captura_real/bunny_x200
  ./run.sh fondo reconstruir real/bunny_x200
  ./run.sh modelo ~/Escritorio/3D_reconstruction/3D_Models/sneaker.usdz
  ./run.sh fondo lote ~/Escritorio/3D_reconstruction/3D_Models --tamano-mm 150
  ./run.sh fondo objetos caesar artefacto_pi2
  ./run.sh fondo pares cesar P1            # real vs gemelo, 4 tecnicas
  PERFIL=rapido ./run.sh reconstruir sintetica/sneaker gaussian
EOF
}

# --- rutas de una captura "real/<x>" o "sintetica/<x>" -------------------------------------------
captura_dir() {
    case "$1" in
        real/*) echo "$ROOT/Captura_real/${1#real/}" ;;
        sintetica/*) echo "$ROOT/Captura_sintetica/${1#sintetica/}" ;;
        *) die "captura '$1': usa real/<nombre> o sintetica/<nombre>" ;;
    esac
}
recon_dir() { echo "$ROOT/Reconstrucciones/$1"; }

# --- marcas de etapa terminada y candado de GPU ---------------------------------------------------
# marca = "<perfil> <fecha>"; una etapa hecha con otro perfil (p. ej. rapido) no cuenta, salvo
# 'entrada' (no depende del perfil)
hecho() {
    local f; f="$(recon_dir "$1")/estado/$2.hecho"
    [ "$FORCE" != 1 ] && [ -f "$f" ] || return 1
    [ "$2" = entrada ] || [ "$(cut -d' ' -f1 "$f")" = "$PERFIL" ]
}
marcar() { mkdir -p "$(recon_dir "$1")/estado"; echo "$PERFIL $(date '+%F %T')" > "$(recon_dir "$1")/estado/$2.hecho"; }
con_gpu() {   # serializa las etapas que usan la GPU (varias terminales / trabajos en fondo)
    exec 9>"$LOGS/.gpu.lock"
    if ! flock -n 9; then log "GPU ocupada por otra etapa: esperando..."; flock 9; fi
    "$@"; local st=$?
    flock -u 9
    return $st
}
etapa() {     # etapa <captura> <nombre> <log> <comando...>: idempotente, con log
    local cap="$1" name="$2" logf="$3"; shift 3
    if hecho "$cap" "$name"; then log "$cap: $name ya hecha (FORCE=1 para rehacer)"; return 0; fi
    mkdir -p "$(dirname "$logf")"
    log "$cap: $name (perfil $PERFIL) -> log $logf"
    local t0=$SECONDS
    if "$@" >"$logf" 2>&1; then
        marcar "$cap" "$name"
        log "$cap: $name OK en $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s"
        grep -E "^-> |^ICP|^GT |^  [a-z]+ +(chamfer|limpieza)|^entrenamiento" "$logf" | sed 's/^/    /'
        return 0   # (sin esto, un grep sin coincidencias + pipefail marcaba la etapa como fallida)
    else
        log "$cap: $name FALLO (ultimas lineas de $logf):"
        grep -v -E "^\s*$|WARN|│|╭|╰|·|⚠|help:|= \"2.30\"" "$logf" | tail -8 | sed 's/^/    /'
        return 1
    fi
}

isaac() { env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONDONTWRITEBYTECODE=1 "$ISAAC/python.sh" "$@"; }

# --- captura sintetica ----------------------------------------------------------------------------
simular() {   # simular <gemelo|modelo> <arg> [opciones]
    local out name logf
    out="$(isaac sim/capture.py "$@" --solo-ruta 2>/dev/null | tail -1)"
    [ -n "$out" ] || die "no pude resolver la salida de sim/capture.py $*"
    name="$(basename "$out")"
    if [ "$FORCE" != 1 ] && [ -f "$out/session.json" ]; then
        log "sintetica/$name ya existe (FORCE=1 para regenerar)"; return 0
    fi
    logf="$LOGS/${name}_captura.log"
    case "$out" in   # regenerar = empezar de cero (sin restos de una captura anterior)
        "$ROOT/Captura_sintetica/"?*) rm -rf "$out" ;;
        *) [ -e "$out" ] && die "--salida fuera de Captura_sintetica/ y ya existe: $out (borrala a mano)" ;;
    esac
    log "sintetica/$name: render en Isaac Sim -> log $logf"
    con_gpu isaac sim/capture.py "$@" >"$logf" 2>&1
    [ -f "$out/ground_truth.json" ] || { log "fallo el render:"; grep -E "Error|error|Traceback" -A3 "$logf" | tail -8; return 1; }
    grep -E "^(gemelo|aleatorio|objeto|[0-9]+ vistas)" "$logf" | sed 's/^/    /'
    log "sintetica/$name: formato real (capture_stop de la captura)"
    tools/cv_py.sh -m sim.write_capture "$out" >>"$logf" 2>&1 || { tail -5 "$logf"; return 1; }
    log "sintetica/$name: validacion"
    local st=0
    tools/cv_py.sh -m eval.validate_geometry "$out" > "$LOGS/${name}_validacion.log" 2>&1 || st=1
    tools/cv_py.sh -m eval.validate_charuco "$out" >> "$LOGS/${name}_validacion.log" 2>&1 || st=1
    tools/cv_py.sh -m eval.compare_format "$out" >> "$LOGS/${name}_validacion.log" 2>&1 || st=1
    grep -E "^resumen|^formato" "$LOGS/${name}_validacion.log" | sed 's/^/    /'
    [ $st -eq 0 ] && log "sintetica/$name lista: $out" \
                  || log "sintetica/$name: HAY VALIDACIONES QUE FALLARON (ver $LOGS/${name}_validacion.log)"
    # las reconstrucciones viejas de esta captura quedan obsoletas
    [ -d "$(recon_dir "sintetica/$name")/estado" ] && rm -f "$(recon_dir "sintetica/$name")"/estado/*.hecho
    return $st
}

lote() {
    local dir="$1"; shift
    [ -d "$dir" ] || die "no existe la carpeta $dir"
    local fallos=()
    for f in "$dir"/*.usd "$dir"/*.usda "$dir"/*.usdc "$dir"/*.usdz; do
        [ -e "$f" ] || continue
        local n; n="$(basename "${f%.*}")"
        simular modelo "$f" "$@" && SIN_INFORME=1 reconstruir "sintetica/$n" || fallos+=("$n")
    done
    informe
    [ ${#fallos[@]} -eq 0 ] && log "lote completo" || log "lote: fallaron ${fallos[*]}"
}

objetos() {   # objetos [claves...]: cada objeto de configs/objects.yaml -> captura sintetica + 4 tecnicas
    local claves=("$@") fallos=()
    if [ ${#claves[@]} -eq 0 ]; then
        mapfile -t claves < <(tools/cv_py.sh -c \
            "from common.config import load_config; print(*load_config('objects')['objects'], sep=chr(10))")
    fi
    log "objetos: ${claves[*]}"
    for k in "${claves[@]}"; do
        simular modelo "$k" && SIN_INFORME=1 reconstruir "sintetica/$k" || fallos+=("$k")
    done
    informe
    [ ${#fallos[@]} -eq 0 ] && log "objetos completos" || { log "objetos: fallaron ${fallos[*]}"; return 1; }
}

pares() {    # pares [capturas reales...]: gemelo + reconstruir sintetica y real + informe (real vs sintetico)
    local caps=("$@") fallos=()
    if [ ${#caps[@]} -eq 0 ]; then
        for d in "$ROOT"/Captura_real/*/session.json; do caps+=("$(basename "$(dirname "$d")")"); done
    fi
    log "pares: ${caps[*]}"
    for c in "${caps[@]}"; do
        c="${c#Captura_real/}"; c="${c%/}"
        # el gemelo va primero: la real usa su GT (altura de la ROI y evaluacion)
        if simular gemelo "Captura_real/$c"; then
            SIN_INFORME=1 reconstruir "sintetica/$c" || fallos+=("sintetica/$c")
        else
            fallos+=("gemelo/$c")
        fi
        SIN_INFORME=1 reconstruir "real/$c" || fallos+=("real/$c")
    done
    informe
    [ ${#fallos[@]} -eq 0 ] && log "pares completos" || { log "pares: fallaron ${fallos[*]}"; return 1; }
}

# --- reconstruccion -------------------------------------------------------------------------------
entrada() {
    local cap="$1" src rd; src="$(captura_dir "$cap")"; rd="$(recon_dir "$cap")"
    [ -f "$src/session.json" ] || die "no existe la captura $src"
    hecho "$cap" entrada && { log "$cap: entrada ya hecha (FORCE=1 para rehacer)"; return 0; }
    etapa "$cap" entrada "$rd/logs/entrada.log" bash -c \
        "tools/cv_py.sh -m eval.export_recon '$src' --check && tools/cv_py.sh -m eval.triangulate '$rd/entrada'" \
        || return 1
    # entrada nueva: las tecnicas hechas con la anterior quedan obsoletas
    for t in $TECNICAS_TODAS evaluar; do rm -f "$rd/estado/$t.hecho"; done
}
tecnica() {
    local t="$1" cap="$2" rd; rd="$(recon_dir "$cap")"
    [ -f "$rd/estado/entrada.hecho" ] || FORCE=0 entrada "$cap" || return 1
    case "$t" in
        profundidad) etapa "$cap" "$t" "$rd/logs/$t.log" tools/ns_py.sh -m recon.profundidad "$rd" ;;
        fotogrametria) etapa "$cap" "$t" "$rd/logs/$t.log" con_gpu tools/ns_py.sh -m recon.fotogrametria "$rd" ;;
        nerf|gaussian) etapa "$cap" "$t" "$rd/logs/$t.log" con_gpu tools/ns_py.sh -m recon.nerfstudio_tecnica "$t" "$rd" ;;
        *) die "tecnica desconocida: $t" ;;
    esac
}
evaluar() {
    local cap="$1" rd; rd="$(recon_dir "$cap")"
    FORCE=1 etapa "$cap" evaluar "$rd/logs/evaluar.log" tools/ns_py.sh -m recon.evaluar "$rd"
}
reconstruir() {
    local cap="$1"; shift
    local ts="${*:-$TECNICAS_TODAS}" fallos=()
    entrada "$cap" || { log "$cap: sin entrada valida, no se reconstruye"; return 1; }
    # la real necesita profundidad para registrar el GT: va primero
    for t in $ts; do tecnica "$t" "$cap" || fallos+=("$t"); done
    evaluar "$cap" || fallos+=(evaluar)
    [ "${SIN_INFORME:-0}" = 1 ] || informe
    [ ${#fallos[@]} -eq 0 ] || { log "$cap: fallaron ${fallos[*]}"; return 1; }
}

limpiar() {
    local rd; rd="$(recon_dir "$1")"
    tools/ns_py.sh -m recon.limpieza "$rd" 2>&1 | grep -E "^  [a-z]+ +ROI"
}
informe() {
    log "informe -> Resultados/evaluacion/informe.md"
    tools/ns_py.sh -m recon.informe >"$LOGS/informe.log" 2>&1 \
        || { log "informe FALLO:"; grep -v -E "WARN|│|╭|╰|·|⚠|help:" "$LOGS/informe.log" | tail -5; return 1; }
}

reporte() {
    log "reporte -> Resultados/reporte/reporte.tex"
    tools/ns_py.sh -m recon.reporte 2>&1 | grep -E "^(->|latexmk|fallo|!|Traceback|  File|[A-Za-z]*Error)" || true
}

articulo() {
    log "articulo -> Articulo/main.pdf"
    tools/ns_py.sh -m recon.articulo 2>&1 | grep -E "^(->|Traceback|  File|[A-Za-z]*Error)" || true
    (cd "$ROOT/Articulo" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1) \
        && log "articulo OK: Articulo/main.pdf" \
        || { log "articulo FALLO (Articulo/main.log):"; grep -A3 "^!" "$ROOT/Articulo/main.log" | head -12; return 1; }
}

estado() { tools/ns_py.sh -m recon.estado 2>/dev/null; }

fondo() {
    local f="$LOGS/fondo_$(date +%Y%m%d_%H%M%S)_$1.log"
    nohup setsid env FORCE="$FORCE" PERFIL="$PERFIL" "$ROOT/run.sh" "$@" >"$f" 2>&1 < /dev/null &
    log "en segundo plano (PID $!): ./run.sh $*"
    log "seguir con: tail -f $f      estado: ./run.sh estado"
}

main() {
    cmd="${1:-ayuda}"; shift || true
    case "$cmd" in
        ayuda|help|-h|--help) ayuda ;;
        gemelo) [ $# -ge 1 ] || die "uso: ./run.sh gemelo Captura_real/<captura>"; simular gemelo "$@" ;;
        modelo) [ $# -ge 1 ] || die "uso: ./run.sh modelo <ruta.usd|clave> [opciones]"; simular modelo "$@" ;;
        lote) [ $# -ge 1 ] || die "uso: ./run.sh lote <carpeta> [opciones]"; lote "$@" ;;
        objetos) objetos "$@" ;;
        pares) pares "$@" ;;
        limpiar) for c in "$@"; do limpiar "$c"; done ;;
        informe) informe ;;
        reporte) reporte ;;
        articulo) articulo ;;
        entrada) for c in "$@"; do entrada "$c"; done ;;
        profundidad|fotogrametria|gaussian|nerf) for c in "$@"; do tecnica "$cmd" "$c"; done ;;
        evaluar) for c in "$@"; do evaluar "$c"; done ;;
        reconstruir|todo) [ $# -ge 1 ] || die "uso: ./run.sh reconstruir <captura> [tecnicas]"; reconstruir "$@" ;;
        fondo) [ $# -ge 1 ] || die "uso: ./run.sh fondo <etapa> [args]"; fondo "$@" ;;
        estado) estado ;;
        pruebas) tools/cv_py.sh -m unittest discover -s tests -t . ;;
        *) ayuda; die "etapa desconocida: $cmd" ;;
    esac
}

# en una sola linea: bash termina aqui aunque el archivo cambie mientras corre un trabajo
main "$@"; exit $?
