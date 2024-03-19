from rembg import remove
from PIL import Image
import glob


def convertTuple(tup):
    # initialize an empty string
    str = ''
    for item in tup:
        str = str + item
    return str


input_paths = sorted(glob.glob("Data/Input/Images_With_BG/*.jpg"))
i = 0

for input_path in zip(input_paths):
    path = convertTuple(input_path)
    input_img = Image.open(path)
    output = remove(input_img)

    output_path = f"Data/Input/Img_Without_BG/Stereo_Image{i}.png"
    output.save(output_path)
    i += 1
