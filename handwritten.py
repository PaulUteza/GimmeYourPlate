import matplotlib.pyplot as plt
import numpy as np
import os
import math
import cv2
from keras.models import model_from_json
import shutil
import itertools
from keras import backend as K
import re
from collections import Counter

letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_classes = len(letters) + 1

word_cfg = {
    'batch_size': 64,
    'input_length': 30,
    'model_name': 'iam_words',
    'max_text_len': 16,
    'img_w': 128,
    'img_h': 64
}

line_cfg = {
    'batch_size': 16,
    'input_length': 98,
    'model_name': 'iam_line',
    'max_text_len': 74,
    'img_w': 800,
    'img_h': 64
}


def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h-old_h)/2), int((new_h-old_h)/2)+old_h
    w1, w2 = int((new_w-old_w)/2), int((new_w-old_w)/2)+old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2,:] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w<target_w and h<target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w>=target_w and h<target_h:
        new_w = target_w
        new_h = int(h*new_w/w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w<target_w and h>=target_h:
        new_h = target_h
        new_w = int(w*new_h/h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        '''w>=target_w and h>=target_h '''
        ratio = max(w/target_w, h/target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    img = cv2.imread(path)
    try:
        img = fix_size(img, img_w, img_h)
    except:
        print(path)
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img /= 255
    return img

def words(text): return re.findall(r'\w+', text.lower())

words_count = Counter(words(open('ocr/big.txt').read()))
checked_word = words(open('ocr/wordlist_mono_clean.txt').read())

def P(word, N=sum(words_count.values())):
    "Probability of `word`."
    return words_count[word] / N

def correction(word):
    "Most probable spelling correction for word."
    if word.lower() in checked_word:
        new_word = word
    else:
        new_word = max(candidates(word, words_count), key=P)
        if word[0].lower()==new_word[0]:
            new_word = list(new_word)
            new_word[0]=word[0]
            new_word = ''.join(new_word)
    return new_word

def correction_list(words):
    res = []
    for word in words:
        if word.lower() in checked_word:
            new_word = word
        else:
            new_word = max(candidates(word), key=P)
            if word[0].lower()==new_word[0]:
                new_word = list(new_word)
                new_word[0]=word[0]
                new_word = ''.join(new_word)
        res.append(new_word)
    return res

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of words_count."
    return set(w for w in words if w in words_count)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate

    return sorted(res, key=lambda entry: entry[0][0])


def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(letters):
            outstr += letters[c]
    return outstr


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)


def predict_image(model_predict, path, is_word):
    if is_word:
        width = word_cfg['img_w']
    else:
        width = line_cfg['img_w']
    img = preprocess(path, width, 64)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decode_label(net_out_value)
    return pred_texts

def make_predict(img):
    with open('anpr_models/handwritten/word_model_predict.json', 'r') as f:
        w_model_predict = model_from_json(f.read())
    w_model_predict.load_weights('anpr_models/handwritten/iam_words--20--1.425.h5')
    img = prepareImg(img, 64)
    img2 = img.copy()
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('tmp/%d.png'%j, wordImg)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image

    cv2.imwrite('ocr/summary.png', img2)
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.imshow(img2)
    imgFiles = os.listdir('tmp')
    imgFiles = sorted(imgFiles)
    pred_line = []
    for f in imgFiles:
        pred_line.append(predict_image(w_model_predict, 'tmp/'+f, True))
    pred_line_spell = correction_list(pred_line)
    shutil.rmtree('tmp/')
    return fig," ".join(pred_line), " ".join(pred_line_spell)