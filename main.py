import os
import requests
import pyfakewebcam
from PIL import Image, ImageDraw, ImageFont
from colour import Color
import tensorflow as tf
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-vh', '--height',     help="Video height (default 640)", default='640', type=int, required=False)
ap.add_argument('-vw', '--width',      help="Video width (default 480)", default='480', type=int, required=False)
ap.add_argument('-wc', '--webcam',     help="Webcam device (default /dev/video0)", default='/dev/video0', type=str, required=False)
ap.add_argument('-fc', '--fakewebcam', help="Fake Webcam device (default /dev/video20)", default='/dev/video20', type=str, required=False)
ap.add_argument('-p',  '--pixelscale', help="Pixel scale (default 0.15)", default='0.15', type=float, required=False)
ap.add_argument('-c',  '--contrast',   help="Contrast adjustment (default 1)", default='1', type=float, required=False)

args = vars(ap.parse_args())

############################################

MODEL = 'deeplabv3_mnv2_pascal_trainval.pb'
SIZE = [args['height'], args['width']]
INPUT_DEVICE = args['webcam']
FAKE_WEBCAM = args['fakewebcam']
SC = args['pixelscale']
GCF = args['contrast']

'''
MODEL = 'deeplabv3_mnv2_pascal_trainval.pb'
SIZE = [640, 480]
FPS = 20
INPUT_DEVICE = '/dev/video0'
FAKE_WEBCAM = '/dev/video20'
SC = 0.15  # pixel sampling rate in width
GCF = 1  # contrast adjustment
'''

############################################

global sess
global detection_graph


def asciiart(img, SC, GCF, color1='black', color2='blue', bgcolor='white'):
    # The array of ascii symbols from white to black
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Load the fonts and then get the the height and width of a typical symbol
    # You can use different fonts here
    font = ImageFont.load_default()
    letter_width = font.getsize("x")[0]
    letter_height = font.getsize("x")[1]

    WCF = letter_height / letter_width

    # open the input file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Based on the desired output image size, calculate how many ascii letters are needed on the width and height
    widthByLetter = round(img.size[0] * SC * WCF)
    heightByLetter = round(img.size[1] * SC)
    S = (widthByLetter, heightByLetter)

    # Resize the image based on the symbol width and height
    img = img.resize(S)

    # Get the RGB color values of each sampled pixel point and convert them to graycolor using the average method.
    # Refer to https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/ to know about the algorithm
    img = np.sum(np.asarray(img), axis=2)

    # Normalize the results, enhance and reduce the brightness contrast.
    # Map grayscale values to bins of symbols
    img -= img.min()
    img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)

    # Generate the ascii art symbols
    lines = ("\n".join(("".join(r) for r in chars[img.astype(int)]))).split("\n")

    # Create gradient color bins
    nbins = len(lines)
    colorRange = list(Color(color1).range_to(Color(color2), nbins))

    # Create an image object, set its width and height
    newImg_width = letter_width * widthByLetter
    newImg_height = letter_height * heightByLetter
    newImg = Image.new("RGBA", (newImg_width, newImg_height), bgcolor)
    draw = ImageDraw.Draw(newImg)

    # Print symbols to image
    leftpadding = 0
    y = 0
    lineIdx = 0
    for line in lines:
        color = colorRange[lineIdx]
        lineIdx += 1

        draw.text((leftpadding, y), line, color.hex, font=font)
        y += letter_height

    # Save the image file
    return  np.asarray(newImg)


def get_frame(cap):
    _, frame = cap.read()

    if crop_camera:
        frame = frame[crop_y:crop_y + SIZE[1], crop_x:crop_x + SIZE[0]]

    mask = None
    while mask is None:
        try:
            mask = get_mask(frame)
        except requests.RequestException:
            print("mask request failed, retrying")

    # post-process mask and frame
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = cv2.erode(mask, np.ones((20,20), np.uint8) , iterations=1)
    frame = cv2.bitwise_not(frame)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.bitwise_not(frame)
    frame = asciiart(frame, SC, GCF)
    frame = cv2.resize(frame, (width, height))
    return frame


def pre_load(modelPath):
    global sess
    global detection_graph

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def=tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(modelPath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)



def get_mask(image):
    global sess
    global detection_graph

    with detection_graph.as_default():
        width = int(image.shape[0])
        height = int(image.shape[1])
        image = cv2.resize(image, (256, 256))
        batch_seg_map = sess.run('SemanticPredictions:0',
                                 feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})

        seg_map = batch_seg_map[0]
        seg_map[seg_map != 15] = 0
        bg_copy = image.copy()
        mask = (seg_map == 15)
        bg_copy[mask] = image[mask]
        seg_image = np.stack((seg_map, seg_map, seg_map),
                             axis=-1).astype(np.uint8)
        gray = cv2.cvtColor(seg_image, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        out = cv2.resize(thresh, (height, width))
        return out

################################################################################################################


pre_load(MODEL)

# setup access to the *real* webcam
cap = cv2.VideoCapture(INPUT_DEVICE)
height, width = SIZE[1], SIZE[0]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# setup the fake camera
fake = pyfakewebcam.FakeWebcam(FAKE_WEBCAM, width, height)

_, frame = cap.read()

crop_camera = False
if frame.shape[0] != SIZE[1] or frame.shape[1] != SIZE[0]:
    crop_camera = True
    crop_y = int((frame.shape[0] - SIZE[1])/2)
    crop_x = int((frame.shape[1] - SIZE[0])/2)
    frame = frame[crop_y:crop_y + SIZE[1], crop_x:crop_x + SIZE[0]]

# frames forever
while True:
    try:
        frame = get_frame(cap)
    except Exception as e:
        print(e)
    # fake webcam expects RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)