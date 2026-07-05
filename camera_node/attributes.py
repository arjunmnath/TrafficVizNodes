import cv2
import numpy as np


class AttributeExtractor:
    def __init__(self):
        # Basic colors in HSV
        self.color_ranges = {
            "red": [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
            "blue": [(100, 100, 100), (130, 255, 255)],
            "green": [(40, 100, 100), (90, 255, 255)],
            "yellow": [(15, 100, 100), (35, 255, 255)],
            "white": [(0, 0, 200), (180, 30, 255)],
            "black": [(0, 0, 0), (180, 255, 50)],
            "grey": [(0, 0, 50), (180, 50, 200)],
        }

    def extract_color(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        best_color = "unknown"
        max_pixels = 0

        for color, ranges in self.color_ranges.items():
            mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
            # handle wrap-around for red
            if len(ranges) == 4:
                mask1 = cv2.inRange(hsv_crop, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv_crop, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_crop, ranges[0], ranges[1])

            count = cv2.countNonZero(mask)
            if count > max_pixels:
                max_pixels = count
                best_color = color

        return best_color

    def extract_type(self, yolo_class_id: int) -> str:
        # COCO mapping
        mapping = {2: "sedan", 3: "motorcycle", 5: "bus", 7: "truck"}
        return mapping.get(yolo_class_id, "unknown")
