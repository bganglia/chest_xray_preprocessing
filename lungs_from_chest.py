from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

model_path = os.path.join(os.path.dirname(__file__), "model.hdf5"
if not os.path.exists(model_path):
urllib.request.urlretrieve("https://github.com/imlab-uiip/lung-segmentation-2d/blob/master/trained_model.hdf5?raw=true",
                           model_path)
lung_finder = load_model(model_path)

def lungs_from_chest(
        radiograph,
        max_value=255,
        thresh=20,
        out_size=None,
        mask_out = False):
    """Crop out everything but the lungs in a chest X-ray."""
    #Note final size
    print(radiograph.shape)
    if out_size is None:
        out_size = radiograph.shape[:2]
    #Identify lungs
    radiograph = cv2.resize(radiograph,(256,256))
    pred_radiograph = np.array([radiograph/max_value])
    if len(pred_radiograph.shape) == 3:
        pred_radiograph = np.expand_dims(pred_radiograph,3)
    print(pred_radiograph.shape)
    lung_confidence = lung_finder.predict(pred_radiograph)[0]
    plt.imshow(np.repeat(lung_confidence,3,2))
    plt.show()
    input()
    #plt.imshow(lung_confidence)
    #Find contours
    mask = (lung_confidence * 255 > 1).astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
    #Find bounds of all contours
    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > thresh:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            points.extend(box)
    points = np.array(points).astype(np.uint8)
    x, y = zip(*points)
    #Optionally mask out
    if mask_out:
        radiograph = radiograph * mask
    #Select only lungs
    cropped = radiograph[min(x):max(x), min(y):max(y)]
    #Resize to correct size
    return cv2.resize(cropped, out_size)
