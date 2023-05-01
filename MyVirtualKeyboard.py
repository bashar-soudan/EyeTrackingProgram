import sys
from PyQt5.QtCore import Qt,QTimer,pyqtSignal
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QFont
from functools import partial
import win32api
import win32con
import win32gui
from pynput.keyboard import Controller,Key
import ctypes
from win32con import IDC_APPSTARTING, IDC_ARROW, IDC_CROSS, IDC_HAND, \
IDC_HELP, IDC_IBEAM, IDC_ICON, IDC_NO, IDC_SIZE, IDC_SIZEALL, \
IDC_SIZENESW, IDC_SIZENS, IDC_SIZENWSE, IDC_SIZEWE, IDC_UPARROW, IDC_WAIT
from win32gui import LoadCursor, GetCursorInfo
from pynput import mouse

class PyKeyboard(QMainWindow):
    hidden = True
    shift = True
    movement=None
    finished= pyqtSignal()
    """PyCalc's View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()
        self.setWindowFlags(
            Qt.WindowDoesNotAcceptFocus
            |Qt.WindowStaysOnTopHint
        )
        self.timer=QTimer()
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        # Set some main window's properties
        self.setWindowTitle('KeyBoard')
        self.setFixedSize(self.screensize[0]-10, int(self.screensize[1]*0.45))
        self.generalLayout = QVBoxLayout()

        self._centralWidget = QWidget(self)
        self._centralWidget.setFocusPolicy(0)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # Create the display and the buttons
        self._createButtons()
        self._connectSignals()
    def _createButtons(self):
        """Create the buttons."""
        self.buttons = {}
        self.ar_buttons={}
        buttonsLayout = QGridLayout()
        # Button text | position on the QGridLayout
        buttons = {'0':(0,0),
                   '1':(0,1),
                   '2': (0, 2),
                   '3': (0, 3),
                   '4': (0, 4),
                   '5': (0, 5),
                   '6': (0, 6),
                   '7': (0, 7),
                   '8': (0, 8),
                   '9': (0, 9),
                   'Q,W': (1, 0),
                   'E,R': (1, 2),
                   'T,Y': (1, 4),
                   'U,I': (1, 6),
                   'O,P': (1, 8),
                   'A,S': (2, 0),
                   'D,F': (2, 2),
                   'G,H': (2, 4),
                   'J,K': (2, 6),
                   'L,M': (2, 8),
                   'To arabic': (3, 0),
                   'Z,X': (3, 2),
                   'C,V': (3, 4),
                   'B,N': (3, 6),
                   'Caps_Lock': (3, 8),
                   'Space': (4,2),
                   'Enter': (4, 4),
                   'BackSpace': (4, 6)
                   }
        ar_buttons = {'0': (0, 0),
                   '1': (0, 1),
                   '2': (0, 2),
                   '3': (0, 3),
                   '4': (0, 4),
                   '5': (0, 5),
                   '6': (0, 6),
                   '7': (0, 7),
                   '8': (0, 8),
                   '9': (0, 9),
                   'ض,ص': (1, 0),
                   'ث,ق': (1, 2),
                   'ف,غ': (1, 4),
                   'ع,ه': (1, 6),
                   'خ,ح,ج': (1, 8),
                   'ش,س,د': (2, 0),
                   'ي,ب,ظ': (2, 2),
                   'ل,ا': (2, 4),
                   'ت,ن,و': (2, 6),
                   'م,ك,ز': (2, 8),
                   'To english': (3, 0),
                   'ط,ئ,ء': (3, 2),
                   'ؤ,ر,ذ': (3, 4),
                   'لا,ى,ة': (3, 6),
                   'Space': (4, 2),
                      'Enter':(4,4),
                      'BackSpace':(4,6)
                   }
        hidden_ar_buttons={'ض': (0, 0),
                        'ص': (0, 1),
                        'ث': (0, 0),
                        'ق': (0, 1),
                        'ف': (0, 0),
                        'غ': (0, 1),
                        'ع': (0, 0),
                        'ه': (0, 1),
                        'خ': (0, 2),
                        'ح': (0, 1),
                        'ج':(0,0),
                        'ش': (0, 2),
                        'س': (0, 1),
                        'د': (0, 0),
                        'ي': (0, 2),
                        'ب': (0, 1),
                        'ظ': (0, 0),
                        'ل': (0, 1),
                        'ا': (0, 0),
                        'ذ': (0, 0),
                        'ت': (0, 2),
                        'ن': (0, 1),
                        'و':(0,0),
                        'م': (0, 2),
                        'ز': (0, 0),
                        'ك': (0, 1),
                        'ط': (0, 2),
                        'ئ': (0, 0),
                        'ء': (0, 1),
                        'ؤ': (0, 2),
                        'ر': (0, 1),
                        'لا': (0, 2),
                        'ى': (0, 1),
                        'ة': (0, 0)
                           }
        hidden_buttons={'Q': (0, 0),
                        'W': (0, 1),
                        'E': (0, 0),
                        'R': (0, 1),
                        'T': (0, 0),
                        'Y': (0, 1),
                        'U': (0, 0),
                        'I': (0, 1),
                        'O': (0, 0),
                        'P': (0, 1),
                        'A':(0,0),
                        'S': (0, 1),
                        'D': (0, 0),
                        'F': (0, 1),
                        'G': (0, 0),
                        'H': (0, 1),
                        'J': (0, 0),
                        'K': (0, 1),
                        'L': (0, 0),
                        'M': (0, 1),
                        'Z': (0, 0),
                        'X': (0, 1),
                        'C':(0,0),
                        'V': (0, 1),
                        'B': (0, 0),
                        'N': (0, 1)}
        # Create the buttons and add them to the grid layout
        for btnText, pos in buttons.items():
            if pos[0]==0:
                self.buttons[btnText] = QPushButton(btnText)
                self.buttons[btnText].setFont(QFont('Arial', 12))
                self.buttons[btnText].setFocusPolicy(0)
                self.buttons[btnText].setFixedSize(int((self.screensize[0]-10)/10), int(self.screensize[1]*0.1))
                buttonsLayout.addWidget(self.buttons[btnText], pos[0], pos[1])
            else:
                self.buttons[btnText] = QPushButton(btnText)
                self.buttons[btnText].setFont(QFont('Arial', 12))
                self.buttons[btnText].setFocusPolicy(0)
                self.buttons[btnText].setFixedSize(int((self.screensize[0]-10)/4), int(self.screensize[1]*0.1))
                buttonsLayout.addWidget(self.buttons[btnText], pos[0], pos[1])
        for btnText, pos in hidden_buttons.items():
            self.buttons[btnText] = QPushButton(btnText)
            self.buttons[btnText].setFont(QFont('Arial', 12))
            self.buttons[btnText].setFocusPolicy(0)
            self.buttons[btnText].setFixedSize(int((self.screensize[0])/2), int(self.screensize[1]/2))
            buttonsLayout.addWidget(self.buttons[btnText], pos[0], pos[1])
        for btnText, pos in ar_buttons.items():
            if pos[0]==0:
                self.ar_buttons[btnText] = QPushButton(btnText)
                self.ar_buttons[btnText].setFont(QFont('Arial', 12))
                self.ar_buttons[btnText].setFocusPolicy(0)
                self.ar_buttons[btnText].setFixedSize(int((self.screensize[0]-10)/10), int(self.screensize[1]*0.1))
                buttonsLayout.addWidget(self.ar_buttons[btnText], pos[0], pos[1])
            else:
                self.ar_buttons[btnText] = QPushButton(btnText)
                self.ar_buttons[btnText].setFont(QFont('Arial', 12))
                self.ar_buttons[btnText].setFocusPolicy(0)
                self.ar_buttons[btnText].setFixedSize(int((self.screensize[0]-10)/4), int(self.screensize[1]*0.1))
                buttonsLayout.addWidget(self.ar_buttons[btnText], pos[0], pos[1])
        for btnText, pos in hidden_ar_buttons.items():
            self.ar_buttons[btnText] = QPushButton(btnText)
            self.ar_buttons[btnText].setFont(QFont('Arial', 12))
            self.ar_buttons[btnText].setFocusPolicy(0)
            self.ar_buttons[btnText].setFixedSize(int((self.screensize[0])/2), int(self.screensize[1]/2))
            buttonsLayout.addWidget(self.ar_buttons[btnText], pos[0], pos[1])
        for btnText, pos in ar_buttons.items():
            self.ar_buttons[btnText].hide()
        for btnText, btn in self.ar_buttons.items():
            self.ar_buttons[btnText].hide()
        self.is_arabic=False
        for btnText, pos in hidden_buttons.items():
            self.buttons[btnText].hide()
        self.hidden=True
        self.buttons['Caps_Lock'].setCheckable(True)
        # self.buttons['To arabic'].setCheckable(True)
        # Add buttonsLayout to the general layout
        self.generalLayout.addLayout(buttonsLayout)


    def CheckScript(self):
        # if(press & Y-self.rect().)
        if (press & (get_current_cursor() == 'IBeam')):
            target_pos_y = int(mouse_y - (self.screensize[1] / 2))
            if (target_pos_y < 0):
                self.move(0, int(self.screensize[1] / 2))
                self.movement = int(self.screensize[1] / 2)
            else:
                self.move(0, 0)
                self.movement = 0
            self.show()
            window = win32gui.FindWindow(None, "KeyBoard")
            ex_style = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE
            win32api.SetWindowLong(window, win32con.GWL_EXSTYLE, ex_style)
        if(self.movement is None):
            pass
        else:
            topY=self.rect().getCoords()[1]+self.movement
            bottomY=self.rect().getCoords()[3]+self.movement
            if(((topY-mouse_y<topY-bottomY-40)|(mouse_y<topY))&(press)&(get_current_cursor() != 'IBeam')):
                self.hide()
            # self.finished.emit()
    def _connectSignals(self):
        """Connect signals and slots."""

        for btnText, btn in self.buttons.items():
            btn.clicked.connect(partial(self._ChangeKeys, btnText))
        for btnText, btn in self.ar_buttons.items():
            btn.clicked.connect(partial(self._ChangeKeys, btnText))
            # btn.clicked.connect(partial(self.setDisplayText, btnText))

    def _ChangeKeys(self, btnText):
        self.keyboard=Controller()

        if btnText == 'Enter':
            self.keyboard.press(Key.enter)
            self.keyboard.release(Key.enter)
            self.close()

        elif btnText == 'BackSpace':
            self.keyboard.press(Key.backspace)
            self.keyboard.release(Key.backspace)
        elif btnText == 'To arabic':
            self.is_arabic=True
            for btnText2, btn in self.buttons.items():
                btn.hide()
            for btnText2,btn in self.ar_buttons.items():
                if (len(btnText2) > 2) | (btnText2 in ['0','1','2','3','4','5','6','7','8','9']):
                    btn.show()
                    self.ar_buttons[btnText2].show()
        elif btnText == 'To english':
            self.is_arabic = False
            for btnText2, btn in self.ar_buttons.items():
                btn.hide()
            for btnText2,btn in self.buttons.items():
                if (len(btnText2) > 2) | (btnText2 in ['0','1','2','3','4','5','6','7','8','9']):
                    btn.show()
        elif btnText == 'Caps_Lock':
            self.shift = not self.shift
        elif btnText == 'Space':
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)
        elif btnText in ['0','1','2','3','4','5','6','7','8','9']:
            self.keyboard.press(btnText)
            self.keyboard.release(btnText)
        elif self.hidden:
            button = btnText.split(',')
            if(not self.is_arabic):
                for btnText2, btn in self.buttons.items():
                    if btnText2 in button:
                        btn.show()
                    else:
                        btn.hide()
            else:
                for btnText2, btn in self.ar_buttons.items():
                    if btnText2 in button:
                        btn.show()
                    else:
                        btn.hide()
            self.hidden = False
        else:
            if(not self.is_arabic):
                for btnText2, btn in self.buttons.items():
                    if (len(btnText2) > 2) | (btnText2 in ['0','1','2','3','4','5','6','7','8','9']):
                        btn.show()
                    else:
                        btn.hide()
            else:
                for btnText2, btn in self.ar_buttons.items():
                    if (len(btnText2) > 2) | (btnText2 in ['0','1','2','3','4','5','6','7','8','9']):
                        btn.show()
                    else:
                        btn.hide()
            self.hidden = True
            if btnText == 'لا':
                self.keyboard.press('ل')
                self.keyboard.release('ل')
                self.keyboard.press('ا')
                self.keyboard.release('ا')
            else:
                if self.shift:
                    self.keyboard.press(btnText.lower())
                    self.keyboard.release(btnText.lower())

                else:
                    self.keyboard.press(btnText)
                    self.keyboard.release(btnText)

    def run(self):
        listener = mouse.Listener(
            on_click=on_click,
        )
        listener.start()
        self.timer.timeout.connect(self.CheckScript)
        # Set the central widget and the general layout
        self.timer.start(30)
    def return_output(self):
        return self.display.text


DEFAULT_CURSORS = {
    LoadCursor(0, IDC_APPSTARTING): 'appStarting',
    LoadCursor(0, IDC_ARROW): 'Arrow', LoadCursor(0, IDC_CROSS): 'Cross',
    LoadCursor(0, IDC_HAND): 'Hand', LoadCursor(0, IDC_HELP): 'Help',
    LoadCursor(0, IDC_IBEAM): 'IBeam', LoadCursor(0, IDC_ICON): 'ICon',
    LoadCursor(0, IDC_NO): 'No', LoadCursor(0, IDC_SIZE): 'Size',
    LoadCursor(0, IDC_SIZEALL): 'sizeAll',
    LoadCursor(0, IDC_SIZENESW): 'sizeNesw',
    LoadCursor(0, IDC_SIZENS): 'sizeNs',
    LoadCursor(0, IDC_SIZENWSE): 'sizeNwse',
    LoadCursor(0, IDC_SIZEWE): 'sizeWe',
    LoadCursor(0, IDC_UPARROW): 'upArrow',
    LoadCursor(0, IDC_WAIT): 'Wait',
}
press=False
mouse_x=0
mouse_y=-1
def get_current_cursor():
    curr_cursor_handle = GetCursorInfo()[1]
    res = DEFAULT_CURSORS.get(curr_cursor_handle, 'None')
    return res


def on_click(x, y, button, pressed):
    global press
    global mouse_x
    global mouse_y
    press=pressed
    mouse_x=x
    mouse_y=y

def activate_keyboard():
    listener = mouse.Listener(
        on_click=on_click,
    )
    listener.start()
    pykey = QApplication(sys.argv)
    while True:

        if(press & (get_current_cursor() == 'IBeam')):
            view_main = PyKeyboard(mouse_y)
            view_main.show()
            window = win32gui.FindWindow(None, "KeyBoard")
            ex_style = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE
            win32api.SetWindowLong(window, win32con.GWL_EXSTYLE, ex_style)
            pykey.exec_()
            pykey.quit()
            print('yes')
