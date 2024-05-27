import streamlit as st
from Object_Detection import Object_Detection
import cv2

import math
import os
import tempfile

from class_names import class_names
import track


class App:

    def __init__(self):
        self.video_path = None
        
    def scale_value(frame, max_frame):
        y = frame * 100 / max_frame
        return y

    def run(self):
        st.title("Object Detection and Tracking With Fusion Algorithm")
        st.markdown("---")
        st.sidebar.title("Settings")

        data_load_state = st.text("You haven't uploaded a video yet!")
        
        # Checkboxes
        use_faster_rcnn = st.sidebar.checkbox("Faster RCNN Kullan")
        use_yolo = st.sidebar.checkbox("YOLO Kullan")
        st.sidebar.markdown("---")
        
        selected_class_ids = []
        
        class_selection = st.sidebar.multiselect("Videoda takip edilmesini istediğiniz nesneleri seçiniz.", class_names, default="person")
        for class_name in class_selection:
            selected_class_ids.append(class_names.index(class_name))
            
        test = st.sidebar.text(selected_class_ids)
        st.sidebar.markdown("---")
        
        # Upload video
        video_file_buffer = st.sidebar.file_uploader("Video yükle", type=["mp4", "avi"])
        tffile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        
        if video_file_buffer:
            my_bar = st.progress(0, text="Video işleniyor...")
            data_load_state.text(tffile.name)
            tffile.write(video_file_buffer.read())
            video = open(tffile.name, 'rb')
            video_bytes = video.read()
            
            st.sidebar.text("Yüklenen video")
            st.sidebar.video(video_bytes)
            
            print(tffile.name)
            stFrame = st.empty()
            st.sidebar.markdown("---")
            
            track_best, track_faster, track_yolo = st.columns(3)
            
            with track_best:
                st.markdown("**Best Tracked Objects**")
                track_best_text = st.text("0")
            
            with track_faster:
                st.markdown("**Tracked Objects By Faster RCNN**")
                track_faster_text = st.text("0")
                
            with track_yolo:
                st.markdown("**Tracked Objects By YOLO**")
                track_yolo_text = st.text("0")
            
            FRAME_WINDOW = st.image([])
                
                
            object_detection = Object_Detection()
            output_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
            st.text(f"hedef dosya: {output_file.name}")
            cap = cv2.VideoCapture(tffile.name)
            cap_settings = track.get_video_settings(cap)
            out = cv2.VideoWriter(
                str(output_file),
                cv2.VideoWriter_fourcc(*'MP4V'),
                cap_settings["fps"],
                (cap_settings["width"], cap_settings["height"])
            )
            
            track.track_objects_from_video(object_detection=object_detection, target_objects=selected_class_ids, capture=cap, out_writer=out, frame_area=FRAME_WINDOW, progress=my_bar, best_count=track_best_text, faster_count=track_faster_text, yolo_count=track_yolo_text, use_yolo=use_yolo, use_faster_rcnn=use_faster_rcnn)
                
            cap.release()
            out.release()
        

       

