"""
MainWindow.py
=============
Ventana principal: un QStackedWidget que alterna entre el menú (A) y el modo 2D (B).
El modo 3D (C) se agregará después.
"""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from src.gui.MenuView import MenuView
from src.gui.View2D import View2D
from src.hmm.Recognizer import Recognizer


class MainWindow(QMainWindow):
    def __init__(self, rec_2d: Recognizer) -> None:
        super().__init__()
        self.setWindowTitle("Reconocedor HMM 2D/3D · Procesos Estocásticos")
        self.resize(1180, 760)

        self.pilas = QStackedWidget()
        self.menu = MenuView(hay_3d=False)
        self.vista_2d = View2D(rec_2d)
        self.pilas.addWidget(self.menu)
        self.pilas.addWidget(self.vista_2d)
        self.setCentralWidget(self.pilas)

        self.menu.abrir_2d.connect(lambda: self.pilas.setCurrentWidget(self.vista_2d))
        self.vista_2d.volver.connect(self._ir_al_menu)

    def _ir_al_menu(self) -> None:
        self.vista_2d.reiniciar()
        self.pilas.setCurrentWidget(self.menu)
