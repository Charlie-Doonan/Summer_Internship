import cv2
import numpy as np

def find_stem(image, conf=0.5):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    clean = np.zeros_like(mask)

    best = None
    best_score = 0

    for c in cnts:

        area = cv2.contourArea(c)
        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(c)

        if w == 0:
            continue

        shape_score = h / w
        size_score = area

        score = shape_score * (size_score / 1000)

        if score > best_score:
            best_score = score
            best = c

    if best is not None:

        x, y, w, h = cv2.boundingRect(best)

        cv2.drawContours(clean, [best], -1, 255, -1)

        cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 3)
        cv2.drawContours(result, [best], -1, (0,0,255), 2)

    return result, clean