"""
main.py
=======
Punto de entrada de la aplicación.

    python main.py

Requiere haber entrenado antes los modelos:
    python -m training.train              (2D)
    python -m training.train --modo 3d    (3D; si no existen, el botón 3D queda deshabilitado)
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from src.gui.MainWindow import MainWindow
from src.gui import styles
from src.hmm.Recognizer import Recognizer

MODELOS_2D = Path(__file__).parent / "models" / "2d" / "hmm_2d.json"
MODELOS_3D = Path(__file__).parent / "models" / "3d" / "hmm_3d.json"


def main() -> None:
    app = QApplication(sys.argv)
    styles.cargar_fuentes()                       # Lexend, incluida en assets/fonts
    app.setStyleSheet(styles.QSS)
    if not MODELOS_2D.exists():
        QMessageBox.critical(None, "Faltan los modelos",
                             f"No se encontró {MODELOS_2D}.\n\n"
                             "Entrena primero con:\n    python -m training.train")
        sys.exit(1)
    rec_3d = Recognizer.load(MODELOS_3D) if MODELOS_3D.exists() else None
    ventana = MainWindow(Recognizer.load(MODELOS_2D), rec_3d)
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
