import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> None:
    pg.setConfigOptions(antialias=False, background="w", foreground="k")
    app = QApplication(sys.argv)
    app.setApplicationName("EEG Video Recorder")
    app.setOrganizationName("eegvid")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
