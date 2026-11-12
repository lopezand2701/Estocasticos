"""
MainWindow.py
=============
Ventana principal: un QStackedWidget que alterna entre
    A. el menú,  B. el modo 2D  y  C. el modo 3D (si hay modelos 3D entrenados).
"""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from src.gui.MenuView import MenuView
from src.gui.View2D import View2D
from src.hmm.Recognizer import Recognizer


class MainWindow(QMainWindow):
    def __init__(self, rec_2d: Recognizer, rec_3d: Recognizer | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Reconocedor HMM 2D/3D · Procesos Estocásticos")
        self.resize(1180, 760)

        self.pilas = QStackedWidget()
        self.menu = MenuView(hay_3d=rec_3d is not None)
        self.vista_2d = View2D(rec_2d)
        self.pilas.addWidget(self.menu)
        self.pilas.addWidget(self.vista_2d)
        self.vista_3d = None
        if rec_3d is not None:
            from src.gui.View3D import View3D
            self.vista_3d = View3D(rec_3d)
            self.pilas.addWidget(self.vista_3d)
            self.menu.abrir_3d.connect(self._ir_al_3d)
            self.vista_3d.volver.connect(self._ir_al_menu)
        self.setCentralWidget(self.pilas)

        self.menu.abrir_2d.connect(lambda: self.pilas.setCurrentWidget(self.vista_2d))
        self.vista_2d.volver.connect(self._ir_al_menu)

    def _ir_al_3d(self) -> None:
        self.pilas.setCurrentWidget(self.vista_3d)
        self.vista_3d.esfera.setFocus()

    def _ir_al_menu(self) -> None:
        self.vista_2d.reiniciar()
        if self.vista_3d is not None:
            self.vista_3d.reiniciar()
            self.vista_3d.b_auto.setChecked(False)
        self.pilas.setCurrentWidget(self.menu)
