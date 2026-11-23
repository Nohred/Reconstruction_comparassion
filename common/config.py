"""Carga de configuracion YAML y resolucion de rutas relativas a la raiz del repo."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name="common"):
    """Lee configs/<name>.yaml (o una ruta explicita)."""
    p = Path(name)
    if not p.suffix:
        p = CONFIG_DIR / f"{name}.yaml"
    return load_yaml(p)


def repo_path(rel):
    """Ruta absoluta para una ruta relativa a la raiz del repo (las absolutas se respetan)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")


def object_config(spec, objects_cfg=None):
    """Config de un objeto: clave de configs/objects.yaml o ruta a un .usd/.usdz.

    Devuelve los defaults con la entrada aplicada encima; `name` = clave o nombre del archivo.
    """
    cfg = objects_cfg or load_config("objects")
    defaults = cfg.get("defaults", {})
    reales = cfg.get("reales") or {}
    if spec in cfg["objects"]:
        obj, name = cfg["objects"][spec], spec
    elif spec in reales:              # objeto de una captura real: entrada base + sus cambios
        ref = reales[spec]
        name = ref["objeto"]
        obj = {**cfg["objects"][name], **{k: v for k, v in ref.items() if k != "objeto"}}
    elif Path(str(spec)).suffix.lower() in USD_SUFFIXES:
        path = Path(spec).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(path)
        # una ruta que ya esta en objects.yaml usa esa entrada (correcciones de unidades/eje)
        known = [k for k, o in cfg["objects"].items() if repo_path(o["usd"]).resolve() == path.resolve()]
        obj, name = ({"usd": str(path)}, path.stem) if not known else (cfg["objects"][known[0]], known[0])
    else:
        raise KeyError(f"'{spec}' no es una clave de configs/objects.yaml {sorted(cfg['objects'])} "
                       f"ni una ruta a un archivo {USD_SUFFIXES}")
    out = dict(defaults)
    out.update({k: v for k, v in obj.items() if k != "placement"})
    out["placement"] = {**defaults.get("placement", {}), **obj.get("placement", {})}
    out["name"] = name
    out["material_por_defecto"] = cfg["material_por_defecto"]
    return out
