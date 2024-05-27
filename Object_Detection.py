import torch
import torchvision
from torchvision import transforms as T

from ultralytics import YOLO


class Object_Detection:
    def faster_rcnn(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision
                                                                     .models
                                                                     .detection
                                                                     .faster_rcnn
                                                                     .FasterRCNN_ResNet50_FPN_Weights
                                                                     .DEFAULT)
        model.eval()
        return model

    def faster_rcnn_predict(self, model, threshold, image_path):
        # img = Image.open(image_path)
        transform = T.ToTensor()
        image = transform(image_path)
        
        with torch.no_grad():
            predictions = model([image])

        bboxes, labels, scores = predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']
        filter_count = torch.argwhere(scores > threshold).shape[0]
        bboxes = bboxes[:filter_count]
        labels = labels[:filter_count]
        scores = scores[:filter_count]
        
        return bboxes, labels, scores

    def yolo(self, yolo_version):
        return YOLO(yolo_version)
        

    def yolo_predict(self, model, threshold, image_path):
        detected_objects = model.predict(source=image_path, save=False)
        bboxes = []
        class_ids = []
        scores = []
        filter_count = 0
        for object in detected_objects:
            bboxes = object.boxes.xyxy
            class_ids = object.boxes.cls
            scores = object.boxes.conf
            filter_count = torch.argwhere(scores > threshold).shape[0]

        return bboxes[:filter_count], class_ids[:filter_count], scores[:filter_count]