"""
main.py
=======
Punto de entrada de la aplicación.

    python main.py

Requiere haber entrenado antes los modelos:
    python -m training.train
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from src.gui.MainWindow import MainWindow
from src.gui.styles import QSS
from src.hmm.Recognizer import Recognizer

MODELOS_2D = Path(__file__).parent / "models" / "2d" / "hmm_2d.json"


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    if not MODELOS_2D.exists():
        QMessageBox.critical(None, "Faltan los modelos",
                             f"No se encontró {MODELOS_2D}.\n\n"
                             "Entrena primero con:\n    python -m training.train")
        sys.exit(1)
    ventana = MainWindow(Recognizer.load(MODELOS_2D))
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
