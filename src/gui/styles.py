"""
styles.py
=========
Identidad visual: "el cuaderno donde piensa el HMM".
Papel frío con renglones, tinta azul para lo que dibuja el usuario,
lápiz rojo para lo que propone el modelo y resaltador para la letra ganadora.
"""
from pathlib import Path

from PySide6.QtGui import QFontDatabase

# ---- paleta ----
PAPEL = "#EEF2F6"        # fondo de la ventana
HOJA = "#FBFCFE"         # tarjetas y lienzo
TINTA = "#1C2F4D"        # trazos del usuario y texto principal
GRAFITO = "#5F7085"      # texto secundario
RENGLON = "#D3DDE8"      # renglones y bordes
RENGLON_FUERTE = "#AFC0D2"
LAPIZ_ROJO = "#D9473F"   # proyección y fase actual del HMM
RESALTADOR = "#FFE07A"   # letra ganadora
NIEBLA = "#E3E9F0"       # superficies secundarias (barras vacías, botones)

FUENTE = "Lexend"
_RAIZ = Path(__file__).resolve().parents[2]


def cargar_fuentes() -> str:
    """Registra Lexend (incluida en assets/fonts, licencia OFL). Si falla, usa la del sistema."""
    for f in sorted((_RAIZ / "assets" / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))
    return FUENTE if FUENTE in QFontDatabase.families() else "Segoe UI"


QSS = f"""
QWidget {{
    background: {PAPEL};
    color: {TINTA};
    font-family: "{FUENTE}", "Segoe UI", sans-serif;
    font-size: 14px;
}}
QLabel {{ background: transparent; }}
QLabel#titulo {{ font-size: 40px; font-weight: 600; letter-spacing: -0.5px; }}
QLabel#bajada {{ color: {GRAFITO}; font-size: 16px; }}
QLabel#seccion {{ font-size: 15px; font-weight: 600; }}
QLabel#nota {{ color: {GRAFITO}; font-size: 12px; }}
QLabel#porcentaje {{ font-size: 30px; font-weight: 600; }}
QLabel#marca {{ font-size: 15px; font-weight: 600; }}

QFrame#hoja {{
    background: {HOJA};
    border: 1px solid {RENGLON};
    border-radius: 18px;
}}
QFrame#hoja QWidget {{ background: transparent; }}
QFrame#barra {{ background: transparent; }}

QPushButton {{
    background: {NIEBLA};
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 14px;
    color: {TINTA};
}}
QPushButton:hover {{ border-color: {RENGLON_FUERTE}; }}
QPushButton:focus {{ border-color: {TINTA}; }}
QPushButton:checked {{ background: {TINTA}; color: {HOJA}; }}
QPushButton:disabled {{ color: {RENGLON_FUERTE}; }}
QFrame#hoja QPushButton {{ background: {NIEBLA}; }}
QFrame#hoja QPushButton:checked {{ background: {TINTA}; color: {HOJA}; }}

QPushButton#segmento {{
    background: transparent; border-radius: 9px; padding: 7px 18px; color: {GRAFITO};
}}
QPushButton#segmento:checked {{ background: {HOJA}; color: {TINTA}; font-weight: 600; }}
QFrame#selector {{ background: {NIEBLA}; border-radius: 12px; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {RENGLON}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {TINTA}; color: {HOJA}; border: none; padding: 6px; }}
"""
