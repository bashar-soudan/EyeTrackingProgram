from re import M
import time
# import win_magnification as mag
from ctypes import *
from tracemalloc import start
import pyautogui
import win_magnification as mag

from pynput import keyboard

factor=1
x_res=1280
y_res=720

start_x=0
start_y=0

def on_press(key):
    global factor
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys'
    print(k)
    if k in ['f5','1']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        if factor<3:    
            factor=factor+0.25
            print(factor)
         # stop listener; remove this if want more keys
    if k in ['f6','2']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        if factor>1: 
            factor=factor-0.25
            print(factor)

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
# listener.join()  # remove if main thread is polling self.keys



def get_Edges(m):
    start_x=m.get_fullscreen_transform()[1][0]
    start_y=m.get_fullscreen_transform()[1][1]

    end_x=int(m.get_fullscreen_transform()[1][0]+x_res/m.get_fullscreen_transform()[0])
    end_y=int(m.get_fullscreen_transform()[1][1]+y_res/m.get_fullscreen_transform()[0])

    return start_x,start_y,end_x,end_y


def isClose(m):
    a,b,c,d=get_Edges(m)

    borderx=50/m.get_fullscreen_transform()[0]
    bordery=50/m.get_fullscreen_transform()[0]

    up=down=right=left=False
    mx,my=pyautogui.position()

    if mx<=a+borderx and mx>borderx:
        left=True
    elif mx>=c-borderx and mx<x_res-borderx:
        right=True
    if my<=b+bordery and my>bordery:
        up=True
    elif my>=d-bordery and my<y_res-bordery:
        down=True

    return up,down,right,left



def start_zoom():
    mag.initialize()
    global start_x,start_y
    while True:
        up,down,right,left=isClose(mag)
        x,y,c,d=get_Edges(mag)

        if up:
            y=y-10
            if y<0:
                y=0
        elif down:
            y=y+10


        if left:
            x=x-10
            if x<0:
                x=0

        elif right:
            x=x+10

        
        if c>x_res:
            x=x-(c-x_res)
        if d>y_res:
            y=y-(d-y_res)

        mag.set_fullscreen_transform(factor,(x,y))

        start_x=x
        start_y=y
        
        pass


# print(mag.get_fullscreen_transform)

# mag.finalize()
