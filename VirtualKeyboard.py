import sys
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from functools import partial
from PyQt5.QtWidgets import QFrame
import win32api
import win32con
import win32gui
import pywintypes
class frame(QFrame):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowFlags(
            Qt.WindowDoesNotAcceptFocus
            | Qt.WindowStaysOnTopHint
        )
        self.setWindowTitle('KeyBoard')
        self.frame = QFrame(self)
        self.frame.setFocusPolicy(Qt.NoFocus)
        self.button = QPushButton("Toggle", self)
        self.button.setFocusPolicy(Qt.NoFocus)
        self.button.setCheckable(True)
        self.button.setChecked(True)
        self.switched = False
        self.vPolicy = None
        # self.button.setStyleSheet("background-color: lightBlue;")
        self.button.setStyleSheet("""
           QPushButton{
               background-color: #BBB;
               border: none;
               padding: 2px;
           }
           QPushButton:checked, QPushButton:hover{
               font-style:italic;
               background-color:lightBlue;
           }""")
def main():
    """Main function."""
    # Create an instance of QApplication
    pycalc = QApplication(sys.argv)
    # print(pycalc.sessionKey())
    # view.show()
    # Show the calculator's GUI
    # view = PyCalcUi()
    # view_main = Main_window(view=view)
    view_main = frame()
    view_main.frame.setFocusPolicy(Qt.NoFocus)
    view_main.show()
    window = win32gui.FindWindow(None, "KeyBoard")
    print(window)
    ex_style = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE
    win32api.SetWindowLong(window, win32con.GWL_EXSTYLE, ex_style)

    # view.show()
    # Create instances of the model and the controller
    # PyCalcCtrl(view=view)
    # Execute calculator's main loop
    pycalc.exec_()
    # c = view.display.text()
    # print('this is c', c)
    sys.exit()
if __name__ == '__main__':
    main()






