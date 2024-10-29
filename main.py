import cv2
import supervision as sv
import os
from polygon_test import LineIntersectionTest

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class YoloObjectDetection:

    def __init__(self, q_img, model, line_zones):

        self.frame = None
        self.q_img = q_img.get()
        self.model = model
        self.line_zones = line_zones
        self.detections = None
        self.count = None

    def predict(self):
        try:

            box_annotator = sv.BoxAnnotator(
                thickness=2,
            )
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=0)

            for result in self.model(source=self.q_img, classes=0, verbose=False):

                self.frame = result.orig_img
                self.detections = sv.Detections.from_ultralytics(result)

                labels = [
                    f"{self.model.names[class_id]}"
                    for box, mask, confidence, class_id, tracker_id, class_name
                    in self.detections
                ]
                box_annotator.annotate(
                    scene=self.frame,
                    detections=self.detections,

                )
                label_annotator.annotate(scene=self.frame, detections=self.detections, labels=labels)
                # return frame, detections
                self.polygon_test()
                return self.frame
        except Exception as er:
            print(er)

    def polygon_test(self):
        try:
            intersect, self.count = LineIntersectionTest(self.detections, self.line_zones).point_line_intersection_test()

            if intersect:
                self.plots()
            else:
                pass
        except Exception as er:
            print(er)

    def plots(self):
        try:
            cv2.putText(
                img=self.frame,
                text=f"TripWire Detected | Person Count: {self.count}",  # Shortened text
                org=(150, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Changed font style
                fontScale=1,  # Adjust font size
                color=(255, 0, 0),
                thickness=2  # Adjust thickness
            )
            cv2.polylines(self.frame, [self.line_zones], True, (0, 0, 255), 4)
        except Exception as er:
            print(er)