import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
import requests
import io
import json
import datetime
import os
from threading import Thread
import threading
import streamlit_authenticator as stauth
import yaml
# cd /Users/nvtiep/Desktop/demo/face-counting/streamLit-cv-mediapipe/; conda activate web; streamlit run face_mesh_app.py
import warnings
warnings.filterwarnings('ignore')

# Resize Images to fit Container
# @st.cache()
@st.cache_data()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)
    resized = cv.flip(resized, 1)
    return resized

def main_gui():
    ## Create Sidebar
    # st.sidebar.title('Điểm danh gương mặt')
    st.sidebar.title('Attendance Checking')
    ## Get Video/Webcam
    stframe = st.empty()
    video = cv.VideoCapture(0)

    kpil1, kpil2, kpil3 = st.sidebar.columns(3)

    with kpil1:
        st.markdown('**Picture**')
        kpil1_texts = [st.markdown('0')]
        kpil1_texts.append( st.markdown('0'))
        kpil1_texts.append( st.markdown('0'))
    with kpil2:
        st.markdown('**Person**')
        kpil2_texts = [st.markdown('0')]
        kpil2_texts.append( st.markdown('0'))
        kpil2_texts.append( st.markdown('0'))
    with kpil3:
        st.markdown('**Days**')
        kpil3_texts = [st.markdown('0')]
        kpil3_texts.append(st.markdown('0'))
        kpil3_texts.append(st.markdown('0'))

    st.markdown('<hr/>', unsafe_allow_html=True)

    name = ""
    ndays = 0

    if os.path.exists('checkin.npy'):
        with open('checkin.npy', 'rb') as f:
            checkin = np.load(f, allow_pickle=True)
            checkin = checkin.tolist()
    else:
        #frame_tocheck
        checkin = dict()

    ## Face Detection
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.85)

    global face_count, frame, frame_toshow, latest_response

    frame = None
    frame_toshow = None
    latest_response = None
    face_count = 0

    def detection_thread_entry():
        global frame, latest_response, face_count
        timestampt = datetime.datetime.now()
        print('Detection thread started')
        while True:
            dtime = (datetime.datetime.now() - timestampt)
            if dtime.total_seconds() < 3:
                continue
            if face_count == 0:
                face_count += 1
                # continue
            timestampt = datetime.datetime.now()

            url = 'https://api.mmlab.uit.edu.vn/face/api/recognize'
            # url = 'http://127.0.0.1:5001/dataset' 
            is_success, buffer = cv2.imencode(".jpg", frame) #encoding
            frame_buf = io.BytesIO(buffer)
            files = {'files': frame_buf}
            latest_response = requests.post(url, files=files, verify=False)
    # st
    detection_api_thread = Thread(target=detection_thread_entry)
    detection_api_thread.start()

    ########### Detection_thread_exit
    # def detection_thread_exit():
    #     global frame, latest_response, face_count
    #     timestampt = datetime.datetime.now()
    #     print('Detection thread started')
    #     while True:
    #         dtime = (datetime.datetime.now() - timestampt)
    #         if dtime.total_seconds() < 3:
    #             continue
    #         if face_count == 0:
    #             continue
    #         timestampt = datetime.datetime.now()

    #         url = 'https://api.mmlab.uit.edu.vn/face/api/recognize'
    #         is_success, buffer = cv2.imencode(".jpg", frame)
    #         frame_buf = io.BytesIO(buffer)
    #         files = {'files': frame_buf}
    #         latest_response = requests.post(url, files = files, verify = False)


    while video.isOpened():
        ret, frame = video.read()
        frame_toshow = frame.copy()
        if not ret:
            continue
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)
        frame.flags.writeable = True

        face_count = 0
        if results.detections:
            frame_toshow = frame.copy()
            #Face Drawing
            for detection in results.detections:
                face_count += 1
                detection.location_data.relative_bounding_box.width = min(detection.location_data.relative_bounding_box.width, 1 - detection.location_data.relative_bounding_box.xmin)
                detection.location_data.relative_bounding_box.height = min(detection.location_data.relative_bounding_box.height, 1 - detection.location_data.relative_bounding_box.ymin)
                mp.solutions.drawing_utils.draw_detection(frame_toshow, detection)
            # Call API
            if latest_response != None:
                persons = json.loads(latest_response.text)
                for idx, person in enumerate(persons):
                    path = 'https://api.mmlab.uit.edu.vn/face/' + person["path"]
                    # path = 'http://127.0.0.1:5001/' + person["signup"]
                    if person["sim"] < 0.6: #person similarity
                        st.sidebar.image(path)
                        # ten = st.sidebar.text_input("Tên")
                        # email = st.sidebar.text_input("Email")
                        # ten = st.text_input("Ten")
                        Name = st.text_input ("Name")
                        email = st.text_input("Email")
                        # dangky = st.sidebar.button("Đăng ký")
                        dangky = st.button("Register")
                        # if st.sidebar.button("Đăng ký"):
                        if st.button("Register"):
                        # if dangky == True:
                            # print(ten)
                            print(Name)
                        continue
                    name = person["name"].split(" - ")[0]
                    now = datetime.datetime.now()
                    if name not in checkin:
                        checkin[name] = dict()
                        checkin[name][now.year] = dict()
                        checkin[name][now.year][now.month] = dict()
                        checkin[name][now.year][now.month][now.day] = [now]
                    elif now.year not in checkin[name]:
                        checkin[name][now.year] = dict()
                        checkin[name][now.year][now.month] = dict()
                        checkin[name][now.year][now.month][now.day] = [now]
                    elif now.month not in checkin[name][now.year]:
                        checkin[name][now.year][now.month] = dict()
                        checkin[name][now.year][now.month][now.day] = [now]
                    elif now.day not in checkin[name][now.year][now.month]:
                        checkin[name][now.year][now.month][now.day] = [now]
                    else:
                        checkin[name][now.year][now.month][now.day].append(now)
                    
                    # Dashboard
                    if name in checkin:
                        ndays = len(checkin[name][now.year][now.month].keys())
                    else:
                        ndays = 0
                    # kpil1_texts[idx] = st.image("path")
                    kpil1_texts[idx].write(f"<img src={path} width=\"56\" height=\"56\"></img>", unsafe_allow_html=True)
                    kpil2_texts[idx].write(f"<p style='text-align: left; color:red;'>{name}</p>", unsafe_allow_html=True)
                    kpil3_texts[idx].write(f"<p style='text-align: left; color:red;'>{ndays}</p>", unsafe_allow_html=True)
                # clear remain
                for idx in range(len(persons), 3):
                    kpil1_texts[idx].write(f"<p style='text-align: left; color:red;'></p>", unsafe_allow_html=True)
                    kpil2_texts[idx].write(f"<p style='text-align: left; color:red;'></p>", unsafe_allow_html=True)
                    kpil3_texts[idx].write(f"<p style='text-align: left; color:red;'></p>", unsafe_allow_html=True)

                with open('checkin.npy', 'wb') as f:
                    np.save(f, checkin)

        frame_toshow = cv.resize(frame_toshow,(0,0), fx=0.8, fy=0.8)
        frame_toshow = image_resize(image=frame_toshow, width=640)
        stframe.image(frame_toshow,channels='BGR', use_column_width=True)


with open('./config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.sidebar.write(f'Welcome *{name}*')
    authenticator.logout('Logout', 'sidebar')
    main_gui()


elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

