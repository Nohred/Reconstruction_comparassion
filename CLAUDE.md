# Gemelo digital de capturas L515 (comparacion de tecnicas de reconstruccion 3D)

Captura real con 3 Intel RealSense L515 fijas (`alta`, `media`, `baja`) y el objeto sobre una
base giratoria con un tablero ChArUco debajo (gira con el objeto). El gemelo en Isaac Sim
reproduce cada captura con ground truth exacto (poses, malla, profundidad, mascaras) y en el
MISMO formato, para correr las mismas tecnicas (COLMAP, NeRF, 3DGS, fusion RGB-D) sobre ambas.

## Estructura

```
Captura_real/<captura>/        ENTRADA real (salida de capture_turntable.py; solo lectura)
Captura_real/scripts/          codigo de captura del colega (solo lectura; se reutiliza por import)
Objects/*.usd                  mallas de los objetos impresos (solo lectura)
Captura_sintetica/<nombre>/    GENERADO (run.sh gemelo|modelo): formato real + ground_truth.json + gt/
Reconstrucciones/<real|sintetica>/<nombre>/
    entrada/                   images/ masks/ sparse/{0,known}/ sparse_pc.ply transforms.json vistas.json
    profundidad/ fotogrametria/ gaussian/ nerf/    resultado.ply (nube, tablero, m), resultado_limpio.ply,
                               malla.ply, info.json
    evaluacion/metricas.json, <tecnica>_error.ply, gt_tablero_m.ply   logs/   estado/*.hecho
Resultados/resumen.csv         tabla de todas las evaluaciones
Resultados/evaluacion/         informe.md + figuras/ (./run.sh informe; generado, no editar)
Resultados/reporte/            reporte.tex/.pdf corto (./run.sh reporte; recon.yaml reporte)
Articulo/                      borrador IEEE TIM (ingles): main.tex + *_texto.tex (a mano), figuras/ tablas/
                               (generadas: ./run.sh articulo, recon.yaml articulo), refs.bib, IEEEtran.cls (CTAN),
                               datos/mediciones_impresion.yaml (calibrador, de mediciones_temp.jpeg)
configs/   common.yaml (rutas, formato, recon/entrada) | twin.yaml (captura sintetica, variabilidad)
           scene.yaml | objects.yaml | recon.yaml (tecnicas, perfiles, evaluacion)
common/    geometria, intrinsecos, lectura de capturas, COLMAP, PLY (numpy)
sim/       capture.py (etapa 1, Isaac) | write_capture.py (etapa 2, OpenCV 4.10) | poses, escena, objetos
eval/      export_recon, triangulate, validate_geometry, validate_charuco, compare_format
recon/     profundidad, fotogrametria, nerfstudio_tecnica (nerf|gaussian), limpieza, evaluar, informe, estado
run.sh     orquestador unico (./run.sh ayuda)
tools/     isaac_py.sh, cv_py.sh, ns_py.sh, inspect/
```

Formato de una captura (real = sintetica): `session.json`, `rig.json`, `<cam>/calib.json`,
`<cam>/stop_XX/{color.png, depth_00..09.png, meta.json}`.

## Flujo

```
./run.sh gemelo Captura_real/bunny_x200           # gemelo exacto (poses de la captura real)
./run.sh modelo <ruta.usd[z]|clave>            # cualquier objeto: plantilla + variabilidad
./run.sh fondo reconstruir <real|sintetica>/<nombre>   # entrada + 4 tecnicas + evaluar + informe (horas)
./run.sh fondo objetos [claves objects.yaml]   # modelo + reconstruir por objeto (Objects/)
./run.sh fondo pares [capturas reales]         # gemelo + reconstruir real y sintetica (real vs gemelo)
./run.sh evaluar <captura> ; ./run.sh informe  # segundos: re-evaluar / Resultados/evaluacion/informe.md
./run.sh reporte                               # Resultados/reporte/reporte.pdf (LaTeX, real vs sintetico)
./run.sh articulo                              # Articulo/main.pdf (figuras/tablas desde las metricas + latexmk)
./run.sh estado
```
Etapas idempotentes (marcas en estado/; FORCE=1 rehace), candado de GPU (logs/.gpu.lock),
PERFIL=rapido para pruebas cortas. **No lanzar etapas largas (nerf, gaussian, fotogrametria,
lote, render) sin que el usuario lo pida.**

## Entornos (dos Python, a proposito)

- Isaac Sim 6.2: `$ISAACSIM_PATH/python.sh` (alias `isaac-python`). Solo para `sim/capture.py`.
  `tools/isaac_py.sh` = Python de Isaac + `pxr` sin SimulationApp (pruebas, inspeccion).
  Correr Isaac con `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV`.
- `tools/cv_py.sh` = conda `twin_cv410` con **OpenCV 4.10.0, el mismo de la captura real**. Toda
  deteccion ChArUco / validacion / exportacion va aqui: el detector de 4.14 (Isaac) desplaza las
  esquinas ~0.4 px respecto a 4.10.
- COLMAP 4.2 con CUDA: conda `colmap_cuda` (`eval.triangulate.colmap_bin`; el `colmap` del entorno
  pixi NO tiene CUDA).
- `tools/ns_py.sh` = entorno pixi de nerfstudio (configs/recon.yaml entornos.nerfstudio_dir):
  nerfstudio 1.1.5, gsplat 1.4, torch 2.2, open3d 0.18, cv2 4.8. gsplat compila con gcc-11
  (CC/CXX los fija ns_py.sh). Ahi corren recon/*.
- Pruebas: `tools/cv_py.sh -m unittest discover -s tests -t .` (tambien con `tools/isaac_py.sh`).

## Convenciones (verificadas con datos, 2026-11-17)

- **Unidades**: poses, rig, mallas GT en **mm**; mundo de Isaac y exportaciones de reconstruccion
  en **metros**; profundidad PNG uint16 con `depth_scale_m` de calib.json (0.00025 m), 0 = invalido.
- **Tablero**: ChArUco 10x7, cuadro 24 mm, marcador 18 mm, DICT_5X5_250 (de `l515_common.py`).
  Hoja carta horizontal; esquina del tablero a (19.67, 7.99) mm de la esquina de la hoja.
- **Marco board** (OpenCV ChArUco): z apunta HACIA LA MESA; las camaras estan en z < 0.
- **meta.json**: `R_board_to_color`, `t_board_to_color_mm` = T_color<-board (p_color = R p_board + t),
  pose FINAL = rig[cam] @ T_rig<-board fusionada. El bloque `directa` es la deteccion propia.
- **rig.json**: T_color<-rig; el rig es el marco de color de la camara de referencia (media).
- **depth_to_color** de calib.json: rotacion column-major (R = reshape(3,3).T), traslacion en m.
  T_depth<-board = inv(T_color<-depth) @ T_color<-board. La profundidad real NO esta alineada.
- **revision/** y la malla GT: p_rev = diag(1, 1, -1) p_board (reflejo, det = -1).
- **Mundo Isaac**: metros, Z arriba, p_world = 0.001 Rx(180) p_board (tablero de la parada base).
  Camara USD mira a -Z: CV_FROM_USD_CAM = diag(1, -1, -1).
- **Gemelo**: camaras FIJAS (rig) y gira la base (`/World/Turntable`: hoja, carton, objeto) con
  T_b0<-bk por parada, reproduciendo exactamente la pose final real (prueba en tests/).
  Mesa, luces y fondo quedan quietos respecto a las camaras, como en el laboratorio.
- **Medio pixel**: el esquema OpenCvPinhole de RTX usa origen en la ESQUINA del pixel; se suma
  `render.opencv_pixel_offset_px: 0.5` a cx, cy solo en la lente (K del GT no cambia).
- **Objetos USD**: declaran metersPerUnit 0.01 / upAxis Y pero son **mm con Z arriba**; se copia
  la malla (no se referencia). `stanford_bunnyx200` = escala fisica. Colocacion: bbox centrado en
  el eje de la base ajustado de las poses (~(121.6, 110.4) mm), yaw 0, apoyado en el tablero.
- **Exportacion recon**: mundo = tablero en metros; images.txt world->cam; transforms.json c2w
  en ejes OpenGL; imagenes sin distorsion (PINHOLE); mascara ROI = cilindro sobre el eje.
- **Centro del pixel**: COLMAP, nerfstudio y gsplat lo ponen en (0.5, 0.5), OpenCV/open3d en (0,0):
  cameras.txt y transforms.json llevan cx, cy + 0.5 (`recon.principal_point_offset_px`);
  vistas.json y las TSDF usan K de OpenCV.
- **Profundidad renderizada**: nerfacto da distancia sobre el rayo (z = d / |(x,y,1)|), splatfacto
  da z; ambas en unidades de nerfstudio (/ scale de dataparser_transforms.json).
- **Captura sintetica `modelo`**: camaras de la plantilla perturbadas una vez por captura; giro
  k*360/N + N(0, 1.22 deg) recortado a 3 deg alrededor del eje ajustado (con su inclinacion).
  rig.json sintetico recalculado. Objeto: metersPerUnit/upAxis del USD salvo correccion en
  objects.yaml; dimension mayor normalizada a 170 mm (tamano_mm / escala). USD con materiales:
  se referencia una version aplanada en Resultados/assets/modelos/<nombre>/ (sim.objects.flatten_asset:
  texturas de USDZ anidados extraidas, pilas de xformOps horneadas —los ops "*:unitsResolve" del
  Metrics Assembler— y SIN fisica: un PhysicsRigidBodyAPI hacia caer el objeto al hacer play()).
  Sin materiales (o --material) se copia la malla con material de config. La captura falla si el
  objeto no aparece en ninguna vista.
- **Evaluacion** (recon/limpieza.py + recon/evaluar.py, configs/recon.yaml limpieza/evaluacion): toda
  nube se limpia igual antes de medir: ROI -> piso (plano RANSAC en la corona exterior de la ROI,
  r > 0.75 R, sin objeto; corte = max(5 mm, grosor del piso p99 + 1 mm), NeRF/3DGS dan piso grueso)
  -> outliers estadisticos -> clusters (semilla = el mayor que TOCA el piso: una sabana de flotantes de
  3DGS supero a bunny_x100; + grandes a < 15 mm; resto flotante). El GT se recorta a la MISMA altura del
  corte de cada tecnica (base no observable). RANSAC con semilla fija.
- **Registro del GT en la real**: ICP de 4 GDL (x, y, z, yaw; apoyado, sin inclinar) sobre CADA tecnica;
  pose = la de menor error mediano en el conjunto (consenso). Gana siempre una RGB: la profundidad L515
  queda ~5-6 mm desplazada del marco ChArUco (16 mm en cesar, PLA glow translucido). Antes se registraba
  sobre profundidad (6 GDL) y P1/cesar daban fitness 0.13-0.18.
- **P1** = pieza propia `Objects/P1.usd` (100x70x33 mm, mm Z arriba), NO la PI2: se detecto porque la nube
  real no ajustaba a la PI2 con ninguna escala/espejo (2026-11-22); el usuario agrego el USD el 2026-11-23.
- **ROI**: altura = max(190, alto GT + 20) mm (caesar mide 200 mm); la real usa el GT de su gemelo.
- **Capturas reales** (2026-11-20, 5): bunny_x200 (= el antiguo `bunnny`, ahora con baja/stop_10,12,14),
  bunny_x100 (stanford_bunny.usd, mitad), cesar (Caesar, PLA glow), artefacto (PI2, PLA beige) y P1 (P1.usd,
  PETG gris). session.json[objeto] -> objeto/material en `configs/objects.yaml: reales`. Materiales
  pla_glow y petg_gris estimados de las fotos (relativos al pla_beige). El gemelo se llama igual que la real.
- **check de entrada**: mediana global de la pose 'directa' solo con vistas de >= 12 esquinas (objetos
  grandes tapan el tablero y el ruido del detector domina).
- **Silueta de validacion**: cv2.fillPoly con muchos poligonos rellena par-impar (solapes se cancelan):
  se rasteriza triangulo por triangulo (eval/validate_geometry.render_silhouette). IoU ~0.99.
- **Validacion ChArUco del gemelo**: criterio global (mediana de todas las vistas y <= 10% de vistas fuera de
  tolerancia, twin.yaml validacion). Antes una sola vista (0.11 vs 0.10 px, u ocluida) tumbaba el gemelo y
  `pares`/`objetos` no lo reconstruian (artefacto y bunny_x100, 2026-11-20).
- **run.sh**: una funcion que termina en `grep | sed` con pipefail devuelve 1 si grep no encuentra
  nada; `etapa` hace `return 0` explicito (eso cortaba `reconstruir` tras una `entrada` nueva).

## Reglas

- **No modificar `Captura_real/` ni `Objects/`** (bloqueado por permisos). Los scripts del colega se
  reutilizan importandolos (`common.board.load_real_script`), con stub de `pyrealsense2`; los cambios
  que necesiten se proponen aparte.
- **Archivos pesados (.ply, .png, .usd, .npy, .npz, .bag) solo con scripts que impriman
  resumenes**: `tools/inspect/run.sh inspect_{ply,png,usd}.py ...`. Nunca `cat`/`head`/`xxd` sobre
  ellos. Para mirar imagenes: `tools/inspect/run.sh make_preview.py a.png b.png --name x` y leer solo
  `tmp/previews/x.jpg` (<= 640 px).
- Configuracion en YAML (`configs/`); sin rutas ni numeros magicos en el codigo.
- **Nunca `cd` a subcarpetas**: mueve el directorio de trabajo que ancla las reglas de permisos.
- Si algo es ambiguo, preguntar.

## Pendiente / limitaciones conocidas

- El conejo real esta ~40 mm fuera del eje; el sintetico va centrado (decision del usuario).
- El gemelo NO reproduce la pose del objeto (colocacion canonica, yaw 0): segun el registro del GT la real
  esta girada ~180 deg (bunny_x100, artefacto, P1), ~90 deg (cesar) y bunny_x200 a yaw 0 desplazado 37 mm.
  Camara/tablero si coinciden. Documentado en el articulo (2026-11-23); para igualarla habria que colocar el
  objeto con registro_gt.T_tablero_m y regenerar el gemelo.
- Ruido de profundidad provisional (gaussiano + perdida + bordes); calibrar mejor.
- Las L515 reales son inconsistentes entre si ~5 mm (pares ICP); el ICP multivista de
  `profundidad` lo corrige (consistencia 1.67 -> 1.07 mm). En sintetico corrige ~0.3 mm.
- Perturbaciones de camara de `modelo` son supuestos (configs/twin.yaml), no medidos.
