import easyocr
import numpy as np
import cv2

class UIDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim'])

    def find_button(self, pil_image, target_text="再来一局"):
        img_array = np.array(pil_image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = self.reader.readtext(img_cv2)
        for (bbox, text, prob) in results:
            if target_text in text.replace(" ", ""):
                (tl, tr, br, bl) = bbox
                cx = int((tl[0] + br[0]) / 2)
                cy = int((tl[1] + br[1]) / 2)
                return (cx/2, cy/2), prob
        return None, 0