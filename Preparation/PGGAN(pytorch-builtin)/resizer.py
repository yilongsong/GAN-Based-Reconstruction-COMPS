#https://github.com/user-cube/Image-Resizer

from PIL import Image
import os
import argparse

def resizeImages():
    """
    Resize a single or multiple images
    and stores them with the choosen
    properties at the selected
    output folder.
    """
    if not os.path.isdir(OUT):
        os.mkdir(OUT)

    if SINGLE == 0:
        isDir = False
    else:
        isDir = True
        entries = os.listdir(CONTENT)

    if isDir:
        for i in entries:
            imageFile = i
            im1 = Image.open(CONTENT + imageFile)
            isOk = False

            if FILTER == 1:
                im2 = im1.resize((WIDTH, HEIGHT), Image.NEAREST)
                isOk = True
            if FILTER == 2:
                im2 = im1.resize((WIDTH, HEIGHT), Image.BILINEAR)
                isOk = True
            if FILTER == 3:
                im2 = im1.resize((WIDTH, HEIGHT), Image.BICUBIC)
                isOk = True
            if FILTER == 4:
                im2 = im1.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
                isOk = True

            if isOk :
                if EXT == '.jpg':
                    im2 =  im2.convert("RGB")

                im2.save(OUT + i.split(".")[0] + EXT)
    else:
        if not os.path.isdir(OUT):
            os.mkdir(OUT)

        im1 = Image.open(CONTENT)
        if FILTER == 1:
            im2 = im1.resize((WIDTH, HEIGHT), Image.NEAREST)
            isOk = True
        if FILTER == 2:
            im2 = im1.resize((WIDTH, HEIGHT), Image.BILINEAR)
            isOk = True
        if FILTER == 3:
            im2 = im1.resize((WIDTH, HEIGHT), Image.BICUBIC)
            isOk =  True
        if FILTER == 4:
            im2 = im1.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            isOk = True

        if isOk :
            if EXT == '.jpg':
                im2 =  im2.convert("RGB")

            im2.save(OUT + CONTENT.split(".")[0] + EXT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", help="Only converts a simple image file", default=0)
    parser.add_argument("--filter", help="1- NEAREST\n2-BILINEAR\n3-BICUBIC\n4-ANTIALIAS", default=1)
    parser.add_argument("--content", help="File or directory to convert", default="myimage.png")
    parser.add_argument("--out", help="Output directory", default="resized/")
    parser.add_argument("--h", help="HEIGHT", default=500)
    parser.add_argument("--w", help="WIDTH", default=500)
    parser.add_argument("--ext", help="Output extension file", default=".png")

    args = parser.parse_args()
    for path in [ "generated.png"]:
        args.content = path
        args.out = "resized/"
        SINGLE = int(args.single)
        FILTER = int(args.filter)
        CONTENT = args.content
        OUT = args.out
        args.h = 1024
        HEIGHT = int(args.h)
        args.w = 1024
        WIDTH = int(args.w)
        EXT = args.ext

        resizeImages()
