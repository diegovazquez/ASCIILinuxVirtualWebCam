import numpy as np
from PIL import Image, ImageDraw, ImageFont
from colour import Color
import cv2


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
    return np.asarray(newImg)

