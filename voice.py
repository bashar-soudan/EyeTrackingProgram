from vosk import Model,KaldiRecognizer
import pyaudio
import pyautogui
import speech_recognition as sr
import requests
import win32gui
def voice_rec(model_c = 'ar'):
    model_en = Model(r"models\vosk-model-small-en-us-0.15")
    model_ar = Model(r"models\vosk-model-ar-mgb2-0.4")
    # Online
    try:
        requests.head('https://google.com/', timeout=1)
        Online=True
    except requests.ConnectionError:
        Online=False
    lang = "en-US"
    if model_c == 'en':
        lang = 'en-US'
    else:
        lang = 'ar-LB'

    if (model_c == 'ar'):
        recognizer = KaldiRecognizer(model_ar, 16000)

    else:
        recognizer = KaldiRecognizer(model_en, 16000)

    if Online == False:
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()

    else:
        r = sr.Recognizer()
        mic = sr.Microphone(device_index=0)

    print("Start")
    print(win32gui.FindWindow(None, "Calib"))
    while True:
        text = ""
        if Online == True:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio, language=lang)
                except Exception as e :
                    print(e)
                print(text)
        else:
            data = stream.read(4096)
            if recognizer.AcceptWaveform(data):
                text = recognizer.Result()[14:-3]
                print(text)

        if text == "open"  or text == "افتح"  :
            xe, ye = pyautogui.position()
            pyautogui.doubleClick(xe,ye)
        if text == "click" or text =="press" or text == "انقر" or text == "اضغط" :
            xe, ye = pyautogui.position()
            pyautogui.click(xe,ye)
        if text == 'close' or text =="اغلق":
            pyautogui.hotkey('altleft', 'f4')

        if text == 'back' or text =="رجوع":
            pyautogui.hotkey('altleft', 'left')

        if text == 'next' or text =="امام":
            pyautogui.hotkey('altleft', 'right')

        if text == 'copy' or text =="انسخ":
            xe, ye = pyautogui.position()
            pyautogui.click(xe,ye)
            pyautogui.hotkey('ctrlleft', 'c')

        if text == 'paste' in text or text =="الصق" or text =="اضع":
            pyautogui.hotkey('ctrlleft', 'v')

        if ('start' in text or text== "ابدا" )and (win32gui.FindWindow(None, "MainWindow") !=0):
            # pyautogui.hotkey('f1')
            pyautogui.keyDown(key='f1')
            pyautogui.keyUp(key='f1')

        if ('setting' in text or 'settings' in text or "اعدادات" in text)and (win32gui.FindWindow(None, "MainWindow") !=0):
            # pyautogui.hotkey('f2')
            pyautogui.keyDown(key='f2')
            pyautogui.keyUp(key='f2')
            return

        if ('calibration' in text or "معايرة" in text)and (win32gui.FindWindow(None, "MainWindow") !=0):
            # pyautogui.hotkey('f3')
            pyautogui.keyDown(key='f3')
            pyautogui.keyUp(key='f3')


        if (text == 'in' or text == "تكبير"):
            # pyautogui.hotkey('f5')
            pyautogui.keyDown(key='f5')
            pyautogui.keyUp(key='f5')

        if ('out' in text or text == "تصغير"):
            # pyautogui.hotkey('f6')
            pyautogui.keyDown(key='f6')
            pyautogui.keyUp(key='f6')

        if (text == 'right click' or text == "ضغط يمين" or text == "right-click"):
            # pyautogui.hotkey('f6')
            print("heheheheheheh")
            pyautogui.rightClick()
