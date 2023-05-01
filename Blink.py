import time

import numpy as np
import cv2 as cv
import pyautogui
from threading import Thread
import os
import timer


from PIL import Image
from face_detection import RetinaFace
from model import L2CS
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
import pickle
import zoom
from sklearn.linear_model import LinearRegression



def goTo(RegModel,frame,Ss):
    tempx=0
    tempy=0

    with torch.no_grad():
        # while True:
        #     ret, frame = cam.read()

            faces = detector(frame)

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

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv.resize(img, (224, 224))
                    # cv.imshow("Resized_Window ", img)
                    # cv.waitKey(1)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

                    result = RegModel.predict([[pitch_predicted, yaw_predicted]])

                    if zoom.factor > 1:
                        start_x = zoom.start_x
                        start_y = zoom.start_y
                        factor = zoom.factor

                        new_xres = Ss[0] / factor
                        new_yres = Ss[1] / factor

                        result[0][0] = start_x + int((new_xres * result[0][0]) / Ss[0])
                        result[0][1] = start_y + int((new_yres * result[0][1]) / Ss[1])
                    return result[0][0],result[0][1]

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



pyautogui.FAILSAFE = False

face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('models/haarcascade_eye_tree_eyeglasses.xml')


def blink(loaded_model,cam,Ss):
    # first_read = True
    # reset = False
    # setup = True
    # blank_image = np.zeros(shape=[22, 22, 3], dtype=np.uint8)
    # cv.namedWindow("Face")
    counter=0
    tempx=0
    tempy=0
    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        gray = cv.bilateralFilter(gray, 5, 1, 1)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        # if (len(faces) > 0):
        #

        for (x, y, w, h) in faces:
                # img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_face = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(30, 30))

                # if (len(eyes) >= 2):
                #     if (first_read) or (reset is False):
                #         cv.putText(img,
                #                    "Eye detected, press S to begin",
                #                    (70, 70),
                #                    cv.FONT_HERSHEY_PLAIN, 3,
                #                    (0, 255, 0), 2)
                #         reset = True

                if (len(eyes) < 2):
                    # if (first_read is False) & (reset is True):
                        # cv.putText(img,
                        #            "Blink detected", (70, 70),
                        #            cv.FONT_HERSHEY_PLAIN, 3,
                        #            (0, 0, 255), 2)
                        xe, ye = pyautogui.position()
                        # pyautogui.doubleClick(xe, ye,button='left')
                        pyautogui.mouseDown(xe, ye)
                        pyautogui.mouseUp(xe, ye)
                        # pyautogui.mouseDown(xe, ye)
                        # pyautogui.mouseUp(xe, ye)
                        time.sleep(1)
                else:
                    counter+=1
                    res=goTo(loaded_model,img,Ss)
                    tempx+=res[0]
                    tempy+=res[1]

                if(counter==3):
                    x=tempx/3
                    y=tempy/3
                    counter=0
                    pyautogui.moveTo(x, y,0.3)
                    tempx=0
                    tempy=0


        # else:
        #     cv.putText(img,
        #                "No face detected", (100, 100),
        #                cv.FONT_HERSHEY_PLAIN, 3,
        #                (0, 255, 0), 2)

        # if setup:
        #     cv.imshow('Face', img)
        #     a = cv.waitKey(1)
        #     if (a == ord('s') and first_read):
        #         first_read = False
        #         setup = False
        #         cv.moveWindow("Face", 1400, 800)

        # else:
        #     cv.imshow('Face', blank_image)
        #     a = cv.waitKey(1)
        #     if (a == ord('q')):
        #         break
            

    # cam.release()
    # cv.destroyAllWindows()



cam_id=0
Screen_size=[]
def start(CameraId,Ss):
    global Screen_size
    global cam_id
    global cam
    global detector
    global transformations
    global gpu
    global model
    global softmax
    global idx_tensor

    Screen_size=Ss
    gpu = select_device("0", batch_size=1)
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    cam = CameraId
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load("models/L2CSNet.pkl")
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    if not cam.isOpened():
        raise IOError("Cannot open webcam")


    loaded_model = pickle.load(open("models/normal.sav", 'rb'))
    t1= Thread(target=zoom.start_zoom)
    t1.start()

    blink(loaded_model,cam,Screen_size)



