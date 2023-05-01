from PyQt5.QtCore import Qt, QThread, pyqtSignal,QTimer
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from face_detection import RetinaFace
from torchvision import transforms

import torch.backends.cudnn as cudnn
from model import L2CS
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import sys
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QMainWindow
import cv2
import ctypes
import os
import timer

Calib_points = [6, 4]
Camera_id = 0
user32 = ctypes.windll.user32
screensize = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv3 🚀  torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')


class Calib(QMainWindow):
    finished = pyqtSignal()

    def __init__(self, Cameraid, Calibpoints, Screensize):
        """View initializer."""
        super().__init__()
        global Calib_points
        global Camera_id
        global screensize
        Calib_points = Calibpoints
        Camera_id = Cameraid
        screensize = Screensize
        cudnn.enabled = True
        self.gpu = select_device("0", batch_size=1)
        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.cam = cv2.VideoCapture(Camera_id)
        self.cam.read()
        print('working')
        self.model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
        saved_state_dict = torch.load("models/L2CSNet.pkl")
        self.model.load_state_dict(saved_state_dict)
        self.model.cuda(self.gpu)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.detector = RetinaFace(gpu_id=0)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(self.gpu)
        self.x = 0
        self.df = pd.DataFrame()
        self.XCounter = 0
        self.YCounter = 0
        self.timer = QTimer()

        # Set some main window's properties
        self.setWindowTitle('Calib')
        self.setFixedSize(int(screensize[0]), int(screensize[1]))
        self.setWindowFlags(Qt.FramelessWindowHint )
        # Set the central widget and the general layout
        self.setStyleSheet("background-color: rgb(37,56,62);")
        self.generalLayout = QVBoxLayout()
        self.setLayout(self.generalLayout)
        self.label_1 = QLabel(self)
        self.label_1.setText('')
        self.label_1.setStyleSheet("border: 3px solid gray;border-radius: 20px;background-color: gray;")
        self.label_1.resize(40, 40)

        # moving position
        self.label_1.move(-20, -20)
        self.label_1.show()
        self.generalLayout.addWidget(self.label_1)

        self.timer.timeout.connect(self.CheckScript)
        self.timer.start(1000)

    def moveto(self):
        if ((self.YCounter == Calib_points[1] - 1) & (self.XCounter == Calib_points[0] - 1)):
            df = self.get_calib_df()
            print(df)
            x = df[['pitch', 'yaw']]
            y = df[['xc', 'yc']]

            linreg = LinearRegression().fit(x, y)

            pickle.dump(linreg, open("models/normal.sav", 'wb'))
            self.cam.release()
            self.close()
            self.finished.emit()

        if (self.XCounter < Calib_points[0] - 1):
            self.XCounter += 1
            self.label_1.move(int(screensize[0] * self.XCounter / (Calib_points[0] - 1)) - 20,
                              int(screensize[1] * self.YCounter / (Calib_points[1] - 1)) - 20)
        else:
            self.XCounter = 0
            self.YCounter += 1
            self.label_1.move(-20, int(screensize[1] * self.YCounter / (Calib_points[1] - 1)) - 20)

    def CheckScript(self):
        res = self.getPoint()
        temp = {'xc': self.label_1.x() + 20, 'yc': self.label_1.y() + 20, 'pitch': res[0], 'yaw': res[1],
                'cx': res[2], 'cy': res[3]}
        self.df = self.df.append(temp, ignore_index=True)
        self.moveto()

    def get_calib_df(self):
        return self.df

    def getPoint(self):
        with torch.no_grad():
            ret, frame = self.cam.read()
            faces = self.detector(frame)

            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    centerx = (x_max + x_min) / 2
                    centery = (y_max + y_min) / 2

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = self.transformations(im_pil)
                    img = Variable(img).cuda(self.gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = self.model(img)

                    pitch_predicted = self.softmax(gaze_pitch)
                    yaw_predicted = self.softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

                    print(pitch_predicted, yaw_predicted)
                    return pitch_predicted, yaw_predicted, centerx, centery

    def run(self):


        self.show()
