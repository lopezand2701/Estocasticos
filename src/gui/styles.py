"""Colores y hoja de estilos (QSS) de toda la aplicación: diseño oscuro y minimalista."""

FONDO = "#11161c"
SUPERFICIE = "#1a222b"
SUPERFICIE_2 = "#222c37"
BORDE = "#2c3a47"
TEXTO = "#e6edf3"
TEXTO_SUAVE = "#8b9bab"
ACENTO = "#3fb7a3"        # verde azulado: trazo y letra ganadora
PROYECCION = "#f2a33a"    # naranja: trayectoria proyectada
GUIA = "#2f3d4a"

QSS = f"""
QWidget {{
    background: {FONDO};
    color: {TEXTO};
    font-family: "Segoe UI", "Inter", "DejaVu Sans", sans-serif;
    font-size: 14px;
}}
QLabel#titulo {{ font-size: 30px; font-weight: 700; }}
QLabel#subtitulo {{ color: {TEXTO_SUAVE}; font-size: 15px; }}
QLabel#seccion {{ color: {TEXTO_SUAVE}; font-size: 12px; font-weight: 600; letter-spacing: 1px; }}
QLabel#prediccion {{ font-size: 64px; font-weight: 700; color: {ACENTO}; }}
QLabel#info {{ color: {TEXTO_SUAVE}; font-size: 12px; }}
QFrame#tarjeta {{
    background: {SUPERFICIE};
    border: 1px solid {BORDE};
    border-radius: 14px;
}}
QFrame#tarjeta QLabel, Canvas2D, ProbabilityPanel,
QFrame#tarjeta QScrollArea, QFrame#tarjeta QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QPushButton {{
    background: {SUPERFICIE_2};
    border: 1px solid {BORDE};
    border-radius: 10px;
    padding: 9px 16px;
}}
QPushButton:hover {{ border-color: {ACENTO}; }}
QPushButton:disabled {{ color: {TEXTO_SUAVE}; border-color: {SUPERFICIE_2}; }}
QPushButton#modo {{ padding: 0; border-radius: 14px; }}
QPushButton#modo QLabel {{ background: transparent; }}
QLabel#modoTitulo {{ font-size: 18px; font-weight: 600; }}
QLabel#modoDesc {{ font-size: 14px; color: {TEXTO_SUAVE}; }}
QLabel#modoTitulo:disabled, QLabel#modoDesc:disabled {{ color: #4a5a69; }}
QPushButton#primario {{ background: {ACENTO}; color: {FONDO}; border: none; font-weight: 600; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDE}; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
