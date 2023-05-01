import sys
from functools import partial
from PyQt5.QtGui import QDoubleValidator,QValidator
from PyQt5.QtWidgets import QErrorMessage
from PyQt5.QtWidgets import QPushButton,QLabel,QLineEdit
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget,QComboBox
from PyQt5.QtCore import QThread,pyqtSignal
import ctypes
import cv2
import wmi
import pythoncom
from math import *


Camera_id=0
CalibX=6
CalibY=4
class Settings(QMainWindow):
    finished = pyqtSignal()
    """PyCalc's View (GUI)."""
    def __init__(self,language,index,calib,calibx,caliby,camid,camindex,screensize,screensizex,screensizey):
        """View initializer."""
        super().__init__()
        self.lang='en'
        user32 = ctypes.windll.user32
        self.screensize = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        # Set some main window's properties
        self.setWindowTitle('Settings')
        # Set the central widget and the general layout
        self.generalLayout = QVBoxLayout()

        self._centralWidget = QWidget(self)
        self._centralWidget.setFocusPolicy(0)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # Create the display and the buttons
        self.Creat_DetectCamera()
        self.text_boxes={}
        self.buttons={}
        self.labels={}
        self.Creat_calib()
        self.Creat_lang()
        button = QPushButton(text="Save settings")
        self.buttons['save_settings'] = button
        self.buttons['save_settings'].clicked.connect(partial(self.SaveSettings,language,index,
                                                              calib,calibx,caliby,camid,camindex,screensize,screensizex,screensizey))
        self.generalLayout.addWidget((button))
    def SaveSettings(self,language,index,calib,calibx,caliby,
                     camid,camindex,screensize,screensizex,screensizey):
        language[index]=self.lang
        calib[calibx]=CalibX
        calib[caliby]=CalibY
        camid[camindex]=Camera_id
        screensize[screensizex]=self.screensize[0]
        screensize[screensizey] = self.screensize[1]
        self.hide()
        # settings.exit()
        self.finished.emit()
        # sys.exit()

    def Creat_DetectCamera(self):
        self.combo1 = QComboBox()
        Cameras = get_cameras()
        if (len(Cameras) == 0):
            self.combo1.addItem('None')
        else:
            for camera in Cameras:
                self.combo1.addItem(camera[0], camera[1])
                self.combo1.activated.connect(self.Apply)
        self.generalLayout.addWidget(self.combo1)


    def Creat_calib(self):
        resolution_layout = QHBoxLayout()
        label1 = QLabel(text='Number of Calibration Points')
        resolution_layout.addWidget(label1)
        textbox1 = QLineEdit()
        textbox1.setPlaceholderText(str(CalibX*CalibY))
        self.text_boxes['Calib'] = textbox1
        resolution_layout.addWidget((textbox1))
        button = QPushButton(text="Confirm")
        self.buttons['Confirm_calib'] = button
        self.buttons['Confirm_calib'].clicked.connect(self.ChangeCalib)
        resolution_layout.addWidget((button))
        self.generalLayout.addLayout(resolution_layout)
    def Creat_lang(self):
        resolution_layout = QHBoxLayout()
        label1 = QLabel(text='the language is English')
        self.labels['language_label']=label1
        resolution_layout.addWidget(label1)
        button = QPushButton(text="Change_language")
        self.buttons['Change_language'] = button
        self.buttons['Change_language'].clicked.connect(self.ChangeLanguage)
        resolution_layout.addWidget((button))
        self.generalLayout.addLayout(resolution_layout)
    def Apply(self,index):
        global Camera_id
        Camera_id=self.combo1.itemData(index)
        print(Camera_id)
    def ChangeResolution(self):
        X=self.text_boxes['ResX'].text()
        Y=self.text_boxes['ResY'].text()
        print(X.isnumeric())
        if (X.isnumeric())&(Y.isnumeric()):
            if(((int(X)>720)&(int(X)<5000)&(int(Y)>480)&(int(Y)<3000))):
                print('correct')
                self.screensize[0] = X
                self.screensize[1] = Y
            else:
                self.text_boxes['ResX'].setText('')
                self.text_boxes['ResY'].setText('')
                error_massage = QErrorMessage()
                error_massage.showMessage('Impossible Resolution')
                error_massage.exec_()

        else:
            self.text_boxes['ResX'].setText('')
            self.text_boxes['ResY'].setText('')
            error_massage=QErrorMessage()
            error_massage.showMessage('Wrong Resolution please try again')
            error_massage.exec_()
    def ChangeCalib(self):
        validation_rule=QDoubleValidator(2,250,0)
        print(validation_rule.validate(self.text_boxes['Calib'].text(),10))
        if(validation_rule.validate(self.text_boxes['Calib'].text(),10)[0]==QValidator.Acceptable):
            n = int(self.text_boxes['Calib'].text())
            for num in range(2, int(n ** 0.5) + 1):
                if n % num == 0:
                    break
                if (n%num != 0) & (num==int(n ** 0.5)):
                    self.text_boxes['Calib'].setText('')
                    error_massage = QErrorMessage()
                    error_massage.showMessage('The Calib Cannot be a Prime Number')
                    error_massage.exec_()
                    return

            val = floor(sqrt(n))
            l2 = []
            for i in range(val, n):
                if n % i == 0:
                    l2.extend([i, n // i])
                    break
            print(l2)
            global CalibX
            global CalibY
            if(l2[0]>l2[1]):
                CalibX=l2[0]
                CalibY=l2[1]
            else:
                CalibX = l2[1]
                CalibY = l2[0]
        else:
            self.text_boxes['Calib'].setText('')
            error_massage = QErrorMessage()
            error_massage.showMessage('Wrong input please try again')
            error_massage.exec_()
    def ChangeLanguage(self):
        if self.lang=='ar':
            self.lang='en'
            self.labels['language_label'].setText('The language is English')
        else:
            self.lang='ar'
            self.labels['language_label'].setText('اللعة عربية')
    def run(self):
        pythoncom.CoInitialize()
        self.show()

def get_cameras():
    index = 0
    arr = []
    c = wmi.WMI()
    Cameras=[]
    wql = "SELECT * FROM Win32_PnPEntity WHERE (PNPClass = 'Image' OR PNPClass = 'Camera')"
    iterator=0
    while True:
        cap = cv2.VideoCapture(index)
        try:
            if cap.getBackendName() == "MSMF":
                arr.append(index)
        except:
            break
        cap.release()
        index += 1
    for item in c.query(wql):
        Cameras.append([item.Name,arr[iterator]])
        iterator+=1
    return Cameras

