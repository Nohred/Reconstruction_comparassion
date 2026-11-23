# Comparación de técnicas de reconstrucción 3D: captura real + capturas sintéticas

Tres L515 fijas, base giratoria con tablero ChArUco y 16 paradas por vuelta. Cada captura, sea real o
sintética, tiene el mismo formato y se reconstruye con las mismas cuatro técnicas y la misma entrada
(poses e intrínsecos conocidos). Así se comparan las técnicas y no la estimación de poses.

```
Captura_real/<captura>/          ENTRADA: salida de capture_turntable.py (solo lectura)
Captura_sintetica/<nombre>/      GENERADO: mismo formato + ground_truth.json y gt/ (malla, profundidad perfecta, máscaras)
Reconstrucciones/<real|sintetica>/<nombre>/
    entrada/                     imágenes sin distorsión, máscaras ROI, COLMAP (sparse/0), transforms.json, vistas.json
    profundidad/                 nubes L515 + ICP multivista + TSDF
    fotogrametria/               COLMAP MVS
    gaussian/                    splatfacto (nerfstudio/gsplat)
    nerf/                        nerfacto (nerfstudio)
                                 cada técnica: resultado.ply (nube cruda), resultado_limpio.ply (la evaluada),
                                 malla.ply, info.json
    evaluacion/metricas.json     limpieza + precisión, completitud, Chamfer, Hausdorff y F-score contra el GT
    evaluacion/<técnica>_error.ply   nube limpia coloreada por su distancia al GT (MeshLab/CloudCompare)
    logs/  estado/
Resultados/resumen.csv           todas las capturas y técnicas en una tabla
Resultados/evaluacion/           informe.md + figuras/ (comparación de todas las capturas evaluadas)
logs/                            capturas sintéticas y trabajos en segundo plano
```

## Uso

Todo pasa por `./run.sh` (`./run.sh ayuda`). Las etapas se saltan si ya están hechas (`FORCE=1` las
rehace) y las que usan la GPU esperan su turno. Las largas se lanzan con `fondo`.

```bash
# 1. Captura real -> su gemelo sintético (mismas poses) y reconstrucción de ambas
./run.sh gemelo Captura_real/bunny_x200
./run.sh fondo reconstruir real/bunny_x200
./run.sh fondo reconstruir sintetica/bunny_x200

# 2. Cualquier modelo USD/USDZ: plantilla de bunny_x200 con variabilidad aleatoria
./run.sh modelo ~/Escritorio/3D_reconstruction/3D_Models/sneaker.usdz          # 170 mm por defecto
./run.sh modelo mi_pieza.usd --tamano-mm 120 --semilla 4 --nombre pieza_s4
./run.sh fondo reconstruir sintetica/sneaker

# 3. Los objetos de configs/objects.yaml (Objects/) o una carpeta completa de modelos
#    (captura + 4 técnicas + evaluación por cada uno, e informe al final)
./run.sh fondo objetos caesar artefacto_pi2      # sin claves = todos los de objects.yaml
./run.sh fondo lote ~/Escritorio/3D_reconstruction/3D_Models

# 4. Solo re-evaluar / regenerar el informe (segundos; p. ej. tras cambiar configs/recon.yaml limpieza)
./run.sh evaluar sintetica/caesar
./run.sh informe                    # -> Resultados/evaluacion/informe.md
./run.sh reporte                    # -> Resultados/reporte/reporte.pdf (LaTeX: tabla detallada real vs sintético)
./run.sh articulo                   # -> Articulo/main.pdf (borrador IEEE TIM; figuras y tablas se regeneran)

./run.sh estado                     # avance y Chamfer por captura y técnica
tail -f logs/fondo_*.log            # seguir un trabajo en segundo plano
PERFIL=rapido ./run.sh reconstruir sintetica/sneaker   # prueba de minutos (pocas iteraciones)
```

Tiempos aproximados por captura (RTX 2000 Ada, perfil `completo`): profundidad ~1 min,
fotogrametría ~20–30 min, gaussian ~20–40 min, nerf ~30–60 min.

## Variabilidad de las capturas sintéticas (`modelo`)

La plantilla es `Captura_real/bunny_x200` (rig, intrínsecos, 16 paradas). En cada captura:
- **Cámaras:** se re-montan una sola vez y quedan fijas durante toda la captura (azimut, elevación,
  distancia, puntería y roll).
- **Base:** gira en cada parada con el ángulo nominal k·22.5° más ruido (σ = 1.2°, máx. 3°, medido en la
  captura real).

La semilla sale del nombre, así que es reproducible; se cambia con `--semilla`. Los parámetros están en
`configs/twin.yaml`.

## Tamaño del objeto

Por defecto la dimensión mayor se normaliza a **170 mm**, como el bunny impreso, para que guarde la misma
proporción con el tablero. Se cambia con `--tamano-mm`, `--escala` o `--tamano-real`, o en
`configs/objects.yaml` por objeto. Los modelos conservan sus materiales y texturas.

## Evaluación contra el ground truth

`./run.sh evaluar <captura>` (también al final de `reconstruir`) compara cada nube con la malla GT, y
`./run.sh informe` junta todas las capturas en `Resultados/evaluacion/informe.md`, con tablas, figuras
de la limpieza y del error por punto. Todas las nubes se tratan igual (`configs/recon.yaml`):

1. **Limpieza** (`recon/limpieza.py`, se guarda en `<técnica>/resultado_limpio.ply`):
   - **ROI:** cilindro sobre el eje de la base. En las sintéticas la altura crece si el objeto es más
     alto, por ejemplo Caesar con 200 mm.
   - **Piso:** el plano del tablero se ajusta con RANSAC en cada nube, sobre la corona exterior de la
     ROI, donde nunca hay objeto. Ahí se mide además el grosor del piso reconstruido, que en NeRF/3DGS
     queda "grueso". Se corta en max(5 mm, grosor + 1 mm).
   - **Outliers estadísticos.**
   - **Clusters flotantes:** se queda el mayor de los que tocan el piso (el objeto está apoyado) y los
     grandes que están a menos de 15 mm de él. El resto son flotantes o restos del piso.
2. **GT:** la malla del objeto en el marco del tablero, menos la franja junto a su base. Esa franja
   no es observable y se excluye a la misma altura del corte del piso de cada técnica. En una captura
   real, la malla del gemelo se registra (ICP de 4 GDL: x, y, z, yaw) sobre cada técnica y se usa la
   pose de consenso (menor error mediano en el conjunto); la misma pose vale para las 4 técnicas.
3. **Métricas (mm):**
   - precisión (nube → malla);
   - completitud (GT → nube);
   - Chamfer, Hausdorff p95;
   - F-score a 1/2/5 mm y su curva;
   - sin ICP (en las poses de la captura) y con ICP (solo error de forma).

Para otro objeto: añadirlo a `configs/objects.yaml` (o pasar la ruta del USD) y correr
`./run.sh fondo objetos <clave>`; la evaluación y el informe salen solos.

## Capturas reales vs su gemelo sintético

`Captura_real/` tiene 5 capturas (bunny_x200, bunny_x100, cesar, artefacto, P1). Cada una se reproduce
con su gemelo (mismas poses de cámara y de la base) y las dos se reconstruyen con las 4 técnicas:

```
./run.sh fondo pares                 # todas las de Captura_real/ (o: ./run.sh fondo pares cesar P1)
```

El objeto y el material del gemelo salen de `session.json["objeto"]` vía `configs/objects.yaml: reales`
(artefacto = PI2 en PLA beige, P1 = `Objects/P1.usd` en PETG gris). Para una captura real nueva: agregar
su `objeto` en `reales` (y el material en `configs/scene.yaml` si hace falta) y correr `pares <captura>`.
El informe agrega la sección *Real vs sintético*: Chamfer y F@2mm real / sintético por técnica, la
diferencia media y `figuras/real_vs_sintetica.png`.

## Usar la entrada con otras herramientas

`Reconstrucciones/<captura>/entrada/` sirve tal cual para:
- **3DGS original:** `-s .../entrada` (lee `images/` y `sparse/0`).
- **nerfstudio:** cualquier método, con `ns-train <método> --data .../entrada nerfstudio-data --eval-mode all`.
- **COLMAP/OpenMVS.**

Marco: tablero ChArUco en metros. Convenciones, entornos y reglas: `CLAUDE.md`.
