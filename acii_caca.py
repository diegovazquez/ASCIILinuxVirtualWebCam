import caca
from caca.canvas import Canvas, CanvasError
from caca.dither import Dither, DitherError
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def asciiart(img, SC, GCF):
    img = cv2.convertScaleAbs(img, alpha=GCF, beta=0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    RMASK = 0x00ff0000
    GMASK = 0x0000ff00
    BMASK = 0x000000ff
    AMASK = 0xff000000
    BPP = 32
    DEPTH = 4

    brightness = None
    contrast = None
    gamma = None
    ditalgo = None
    exformat = "svg"
    charset = None

    width = int(img.size[1] * 2.23 * SC)
    height = int(img.size[0] * 0.75 * SC)

    cv = Canvas(width, height)
    cv.set_color_ansi(caca.COLOR_DEFAULT, caca.COLOR_TRANSPARENT)

    try:
        # convert rgb to rgba
        if img.mode == 'RGB':
            img = img.convert('RGBA')
        # reorder rgba
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = Image.merge("RGBA", (b, g, r, a))
        dit = Dither(BPP, img.size[0], img.size[1], DEPTH * img.size[0], RMASK, GMASK, BMASK, AMASK)
    except DitherError as err:
        print(err)


    if ditalgo:
        dit.set_algorithm(ditalgo)

    # set brightness
    if brightness:
        dit.set_brightness(brightness)

    # set gamma
    if gamma:
        dit.set_gamma(gamma)

    # set contrast
    if contrast:
        dit.set_contrast(contrast)

    # set charset
    if charset:
        dit.set_charset(charset)

    #create dither
    dit.bitmap(cv, 0, 0, width, height, img.tobytes())

    asciiArt = cv.export_to_memory(exformat)

    ##############################################################

    bgcolor = 'white'

    font = ImageFont.truetype("DejaVuSansMono.ttf", 11)

    #letter_width = font.getsize("X")[0]
    letter_height = font.getsize("X")[1]

    lines = asciiArt.split('\n')

    canvasHeight = int(lines[1].split('"')[3])
    canvasWidth = int(lines[1].split('"')[1])

    newImg = Image.new("RGB", (canvasWidth, canvasHeight), bgcolor)
    draw = ImageDraw.Draw(newImg)

    for char in lines[3:-3]:
        type = char.split('"')[0].split(" ")[0][1:]

        x = int(char.split('"')[3])
        y = int(char.split('"')[5])
        color = char.split('"')[1].split(":")[1]
        if type == "text":
            character = str(char.split('"')[-1][1:2])
            draw.text((x, y - letter_height), character, color, font=font)
        else:
            rectWidth = int(char.split('"')[-4])
            rectheight = int(char.split('"')[-2])
            draw.rectangle(((x, y), (x + rectWidth, y + rectheight)), fill=color)

    finalImg = np.asarray(newImg)
    finalImg = cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB)
    return finalImg

'''
img = cv2.imread("04_input_mask.png")
finalImg = asciiart(img, 0.2, 3)
cv2.imwrite("test.png", finalImg)
'''

