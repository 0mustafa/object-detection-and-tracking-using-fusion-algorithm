import cv2
import math


def get_video_settings(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}

def draw_one_model_objects(frame, model_pixels, circle_color, rectangle_color):
    for object_pixels in model_pixels.items():
        x1, y1, x2, y2, center_x, center_y, score = object_pixels[1]
        cv2.circle(frame, (center_x, center_y), 4, circle_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rectangle_color, 2)
        # cv2.putText(frame, str(round(score, 2)), (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA, False)
    return frame

def draw_fusion_objects(frame, best_pixels, model1_pixels, model2_pixels, best_ccolor, best_rcolor, model1_ccolor, model1_rcolor, model2_ccolor, model2_rcolor):
    frame = draw_one_model_objects(frame, model1_pixels, model1_ccolor, model1_rcolor)
    frame = draw_one_model_objects(frame, model2_pixels, model2_ccolor, model2_rcolor)
    frame = draw_one_model_objects(frame, best_pixels, best_ccolor, best_rcolor)

    return frame

def detect_objects(bboxes, labels, scores, target_objects):
    objects = {}
    detected_objects = 0
    centers = {}
    for i, bbox in enumerate(bboxes):
        # Only persons
        if labels[i].item() not in target_objects:
            continue
        detected_objects += 1
        objects[detected_objects] = f"{labels[i].item()}, {scores[i].item()}"
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        center_x = int((x1 + x2)/2)
        center_y = int((y1 + y2)/2)
        centers[detected_objects] = [x1, y1, x2, y2, center_x, center_y, scores[i].item()]

    return objects, detected_objects, centers

def track_objects_from_video(object_detection, target_objects, capture, out_writer, frame_area, progress, best_count, faster_count, yolo_count, use_yolo=True, use_faster_rcnn=True):
    if use_faster_rcnn:
        model_faster = object_detection.faster_rcnn()

    if use_yolo:
        model_yolo = object_detection.yolo("yolov8n.pt")
        
    # videonun özellikleri
    capture_settings = get_video_settings(capture)

    frame_num = 0
    while True:
        ret, frame = capture.read()

        # Okunacak frame kalmadıysa döngüyü sonlandır
        if not ret:
            break

        frame_num += 1
        # Use Fusion Algorithm
        if use_faster_rcnn and use_yolo:
            # Faster RCNN
            f_bboxes, f_labels, f_scores = object_detection.faster_rcnn_predict(model_faster, threshold=0.75, image_path=frame)
            objects_faster_rcnn, detected_object_count_faster_rcnn, centers_faster = detect_objects(f_bboxes, f_labels, f_scores, target_objects=[id+1 for id in target_objects])

            # YOLO
            y_bboxes, y_labels, y_scores = object_detection.yolo_predict(model=model_yolo, threshold=0.38, image_path=frame)
            objects_yolo, detected_object_count_yolo, centers_yolo = detect_objects(y_bboxes, y_labels, y_scores, target_objects=target_objects)
            
            centers_best = {}
            centers_faster_copy = centers_faster.copy()
    
            # Fusion ALgorithm
            for i, vals_faster in enumerate(centers_faster.items()):
                x1_f, y1_f, x2_f, y2_f, center_x_f, center_y_f, score_f = vals_faster[1]
                min_distance = [0,100]
                min_distance = {"dict_yolo_key": 0, "distance": 100}
                for j, vals_yolo in enumerate(centers_yolo.items()):
                    x1_y, y1_y, x2_y, y2_y, center_x_y, center_y_y, score_y = vals_yolo[1]
                    distance = math.dist([center_x_f,center_y_f], [center_x_y, center_y_y])
                    if distance <= min_distance["distance"]:
                        min_distance = {"dict_yolo_key": vals_yolo[0], "distance": distance}
                if min_distance["distance"] <= 15:
                    x1_y, y1_y, x2_y, y2_y, center_x_y, center_y_y, score_y = centers_yolo.get(min_distance["dict_yolo_key"])
                    if score_f > score_y:
                        centers_best[i] = [x1_f, y1_f, x2_f, y2_f, center_x_f, center_y_f, score_f]
                    else:
                        centers_best[i] = [x1_y, y1_y, x2_y, y2_y, center_x_y, center_y_y, score_y]
                    centers_faster_copy.pop(vals_faster[0])
                    centers_yolo.pop(min_distance["dict_yolo_key"])
            # Draw objects detected by YOLO and Faster RCNN
            frame = draw_fusion_objects(frame, centers_best, centers_faster, centers_yolo, (255,0,0), (255,0,0), (0,200,100), (0,255,0), (0,255,255), (0,0,255))
            faster_count.text(f"{detected_object_count_faster_rcnn}")
            yolo_count.text(f"{detected_object_count_yolo}")
            best_count.text(f"{detected_object_count_yolo+detected_object_count_faster_rcnn-len(centers_best)}")
            
            # Yazılacak metin ve konumu
            text = f"Faster RCNN: {detected_object_count_faster_rcnn} YOLO: {detected_object_count_yolo}"
            text2 = f"Frame: {frame_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (200, 200, 0)  # Yazı rengi
            thickness = 3
        
            # Metnin boyutlarını al ve sağ alt köşeye yerleştir
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
            text_x = capture_settings["width"] - text_size[0] - 10  # Sağdan 10 piksel boşluk
            text2_x = capture_settings["width"] - text_size2[0] - 10
            text_y = capture_settings["height"] - 10  # Alttan 10 piksel boşluk
            text2_y = text_y - 40
        
            # Metni frame üzerine yaz
            cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, color, thickness)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
            
        # If only use Faster RCNN
        elif use_faster_rcnn:
            # Faster RCNN
            f_bboxes, f_labels, f_scores = object_detection.faster_rcnn_predict(model_faster, threshold=0.75, image_path=frame)
            objects_faster_rcnn, detected_object_count_faster_rcnn, centers_faster = detect_objects(f_bboxes, f_labels, f_scores, target_objects=[id+1 for id in target_objects])
            # Draw objects detected by Faster RCNN
            frame = draw_one_model_objects(frame, centers_faster, (0,200,100), (0,255,0))
            faster_count.text(f"{detected_object_count_faster_rcnn}")

            # Yazılacak metin ve konumu
            text = f"Faster RCNN: {detected_object_count_faster_rcnn}"
            text2 = f"Frame: {frame_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (200, 200, 0)  # Yazı rengi
            thickness = 3
        
            # Metnin boyutlarını al ve sağ alt köşeye yerleştir
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
            text_x = capture_settings["width"] - text_size[0] - 10  # Sağdan 10 piksel boşluk
            text2_x = capture_settings["width"] - text_size2[0] - 10
            text_y = capture_settings["height"] - 10  # Alttan 10 piksel boşluk
            text2_y = text_y - 40
        
            # Metni frame üzerine yaz
            cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, color, thickness)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
            print(f"video isleniyor... {frame_num}/{capture_settings['total_frames']}")
            
        # If only use YOLO
        elif use_yolo:
            # YOLO
            y_bboxes, y_labels, y_scores = object_detection.yolo_predict(model=model_yolo, threshold=0.38, image_path=frame)
            objects_yolo, detected_object_count_yolo, centers_yolo = detect_objects(y_bboxes, y_labels, y_scores, target_objects=target_objects)
            # Draw objects detected by YOLO
            frame = draw_one_model_objects(frame, centers_yolo, (0,255,255), (0,0,255))
            yolo_count.text(f"{detected_object_count_yolo}")

            # Yazılacak metin ve konumu
            text = f"YOLO: {detected_object_count_yolo}"
            text2 = f"Frame: {frame_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (200, 200, 0)  # Yazı rengi
            thickness = 3
        
            # Metnin boyutlarını al ve sağ alt köşeye yerleştir
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
            text_x = capture_settings["width"] - text_size[0] - 10  # Sağdan 10 piksel boşluk
            text2_x = capture_settings["width"] - text_size2[0] - 10
            text_y = capture_settings["height"] - 10  # Alttan 10 piksel boşluk
            text2_y = text_y - 40
        
            # Metni frame üzerine yaz
            cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, color, thickness)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        # İşlenen kareyi yazma
        out_writer.write(frame)
        # print(f"video isleniyor... {frame_num}/{capture_settings['total_frames']}")
        frame_area.image(frame)
        progress.progress(int((frame_num * 100) / capture_settings["total_frames"]), text=f"Video işleniyor... ({frame_num}/{capture_settings['total_frames']})")
        

    # capture.release()
    # out_writer.release()
    # cv2.destroyAllWindows()