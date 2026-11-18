"""Salida compacta compartida por los scripts de inspeccion (limite de lineas)."""
import sys

DEFAULT_MAX_LINES = 30


class Out:
    """Acumula lineas y nunca imprime mas de `max_lines` (avisa cuantas omitio)."""

    def __init__(self, max_lines=DEFAULT_MAX_LINES):
        self.max_lines = max_lines
        self.n = 0
        self.dropped = 0

    def __call__(self, *parts):
        if self.n < self.max_lines - 1:
            print(*parts)
            self.n += 1
        else:
            self.dropped += 1

    def close(self):
        if self.dropped:
            print(f"... ({self.dropped} lineas omitidas; usa --max-lines o filtra la entrada)")
        sys.stdout.flush()


def add_max_lines(ap):
    ap.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES,
                    help=f"maximo de lineas impresas (default {DEFAULT_MAX_LINES})")


def fmt3(v, nd=4):
    return "(" + ", ".join(f"{float(x):.{nd}g}" for x in v) + ")"
