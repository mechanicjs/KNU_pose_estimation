import time
import cv2
import numpy as np
import ctypes
import _ctypes
import sys
from acquisitionKinect import AcquisitionKinect as ack
from frame import Frame
import cv2
import pygame
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

kinect = ack()
frame = Frame()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        while True:
            kinect.get_frame(frame)
            kinect.get_color_frame()
            image = kinect._frameRGB
            kinect.get_depth_frame()
            image2 = kinect._frameDepth

            image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('live_detection', image_np)
            # cv2.imshow('depth_image', image2) #depth

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                cv2.destroyAllWindows()
                cap.release()