"""
MainWindow.py
=============
Ventana principal: un QStackedWidget que alterna entre
    A. el menú,  B. el modo 2D (plano)  y  C. el modo 3D (esfera, si hay modelos 3D).
Desde B y C se puede pasar directamente al otro modo con el selector Plano/Esfera.
"""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from src.gui.MenuView import MenuView
from src.gui.View2D import View2D
from src.hmm.Recognizer import Recognizer


class MainWindow(QMainWindow):
    def __init__(self, rec_2d: Recognizer, rec_3d: Recognizer | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Reconocedor de letras con cadenas ocultas de Markov")
        self.resize(1280, 800)
        self.setMinimumSize(1100, 700)

        hay_3d = rec_3d is not None
        self.pilas = QStackedWidget()
        self.menu = MenuView(hay_3d=hay_3d)
        self.vista_2d = View2D(rec_2d, hay_3d=hay_3d)
        self.pilas.addWidget(self.menu)
        self.pilas.addWidget(self.vista_2d)
        self.vista_3d = None
        if hay_3d:
            from src.gui.View3D import View3D
            self.vista_3d = View3D(rec_3d, hay_3d=True)
            self.pilas.addWidget(self.vista_3d)
            self.vista_3d.volver.connect(self._ir_al_menu)
            self.vista_3d.cambiar_modo.connect(self._ir_a)
        self.setCentralWidget(self.pilas)

        self.menu.abrir_2d.connect(lambda: self._ir_a("2d"))
        self.menu.abrir_3d.connect(lambda: self._ir_a("3d"))
        self.vista_2d.volver.connect(self._ir_al_menu)
        self.vista_2d.cambiar_modo.connect(self._ir_a)

    def _ir_a(self, modo: str) -> None:
        vista = self.vista_3d if modo == "3d" and self.vista_3d is not None else self.vista_2d
        self.pilas.setCurrentWidget(vista)
        vista.lienzo.setFocus()

    def _ir_al_3d(self) -> None:                # usado por versiones anteriores
        self._ir_a("3d")

    def _ir_al_menu(self) -> None:
        self.vista_2d.reiniciar()
        if self.vista_3d is not None:
            self.vista_3d.reiniciar()
            self.vista_3d.b_auto.setChecked(False)
        self.pilas.setCurrentWidget(self.menu)
