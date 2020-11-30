import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder


def image_processing(LpImg):
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg, alpha=(255.0))

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray ,(7 ,7) ,0)

    # Applied inversed thresh_binary
    binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    return plate_image, binary, dilation


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts



def segment_letters(binary, plate_image, dilatation):
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate

                # Sperate number and gibe prediction
                curr_num = dilatation[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))
    return crop_characters

def load_model(path):
    with open(path+'.json') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights(path+'.h5')
    print("Loading model successfully...")
    labels = LabelEncoder()
    labels.classes_ = np.load(path+'.npy')
    print("Labels loaded successfully...")
    return model, labels


def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


def make_predictions(img):
    plate_image, binary, dilatation = image_processing(img)
    segmented_ch = segment_letters(binary, plate_image, dilatation)
    model, labels = load_model('anpr_models/ocr_mobile/mobile')
    fig = plt.figure(figsize=(15, 3))
    cols = len(segmented_ch)
    grid = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)
    final_string = ''
    for i, character in enumerate(segmented_ch):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character, model, labels))
        plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        plt.axis(False)
        plt.imshow(character, cmap='gray')
    return final_string, fig

