"""
a tool to explore the image in VOC2007 with bbox
"""
import matplotlib.pyplot as plt
import matplotlib.image as imglib
import xml.etree.ElementTree as ET
import sys
import os
data_root = "/2proj/github/py-faster-rcnn/data/VOCdevkit2007/VOC2007"
imgext = ".jpg"
idx = "004686"
if(len(sys.argv)>1):
    idx = sys.argv[1]
imgpath = os.path.join(data_root,"JPEGImages",idx+imgext)
annotpath = os.path.join(data_root,"Annotations",idx+".xml")
assert os.path.exists(imgpath)
assert os.path.exists(annotpath)
parser = ET.parse(annotpath)
objs = parser.findall("object")
boxes = []
for obj in objs:
    bndbox = obj.find("bndbox")
    name = obj.find("name").text
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    ymax = bndbox.find("ymax").text
    xmax = bndbox.find("xmax").text
    box = [float(xmin),float(ymin),float(xmax),float(ymax),name]
    boxes.append(box)
    print box

img = imglib.imread(imgpath)

fig = plt.figure()
ax = fig.add_subplot(111)
for box in boxes:
    ax.add_patch(plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False))
    ax.text(box[0],box[1]-2,box[4])

plt.imshow(img)
plt.show()




