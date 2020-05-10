import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
import pyfakewebcam
import tensorflow as tf
import cv2
import numpy as np
import argparse
import pathlib


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ap = argparse.ArgumentParser()
ap.add_argument('-vh', '--height',     help="Video height (default 640)", default='640', type=int, required=False)
ap.add_argument('-vw', '--width',      help="Video width (default 480)", default='480', type=int, required=False)
ap.add_argument('-wc', '--webcam',     help="Webcam device (default /dev/video1)", default='/dev/video1', type=str, required=False)
ap.add_argument('-fc', '--fakewebcam', help="Fake Webcam device (default /dev/video20)", default='/dev/video20', type=str, required=False)
ap.add_argument('-p',  '--pixelscale', help="Pixel scale (default 0.15)", default='0.1', type=float, required=False)
ap.add_argument('-c',  '--contrast',   help="Contrast adjustment (default 1)", default='1', type=float, required=False)
ap.add_argument('-u',  '--usecaca',    help="Use libcaca for ASCII transformation", default='false', nargs='?', const=True, type=str2bool, required=False)
ap.add_argument('-bg', '--background', help="Background image path", default='background.jpeg', type=str, required=False)

args = vars(ap.parse_args())

############################################

MODEL = 'deeplabv3_mnv2_pascal_trainval.pb'
SIZE = [args['height'], args['width']]
INPUT_DEVICE = args['webcam']
FAKE_WEBCAM = args['fakewebcam']
SC = args['pixelscale']
GCF = args['contrast']
BGIMGPATH = args['background']

############################################

file = pathlib.Path(BGIMGPATH)
if file.exists():
    backgroundImage = cv2.imread(BGIMGPATH)
else:
    backgroundImage = False
    print("Background not found")

############################################
global sess
global detection_graph


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
    mask = cv2.erode(mask, np.ones((20, 20), np.uint8), iterations=1)

    if file.exists():
        alpha = mask.astype(float) / 255
        foreground = cv2.multiply(alpha, frame, dtype=cv2.CV_32F)
        background = cv2.multiply(1.0 - alpha, backgroundImage, dtype=cv2.CV_32F)
        frame = cv2.add(foreground, background)
        frame = np.uint8(frame)
    else:
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

print("Model File\t\t" + str(MODEL))
print("Camera Size\t\t" + str(SIZE))
print("Input Webcam\t\t" + str(INPUT_DEVICE))
print("Fake Webcam Dev\t\t" + str(FAKE_WEBCAM))
print("Scale\t\t\t" + str(SC))
print("Contrast\t\t" + str(GCF))
print("Use caca\t\t" + str(args['usecaca']))
print("Bg Image\t\t" + str(BGIMGPATH))


if args['usecaca']:
    from acii_caca import asciiart
else:
    from acii import asciiart

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