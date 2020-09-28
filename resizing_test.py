import cv2
import matplotlib.pyplot as plt
import numpy as np


def show(image, min, max, title):
    plt.imshow(image, vmin=min, vmax=max, cmap='plasma')
    plt.title(title)
    plt.colorbar()
    plt.show()


shape_input = (288, 512)

# set min and max
max_depth = 100
min_depth = 5

# load depthmap and clip to min_depth/max_depth
yaml_file = cv2.FileStorage('samples/73_10.yml', cv2.FILE_STORAGE_READ)
depthmap = yaml_file.getNode('depthmap_73_10').mat()
print('original min max:', np.amin(depthmap), np.amax(depthmap))
depthmap = np.clip(depthmap, min_depth, max_depth)
show(depthmap, np.amin(depthmap), np.amax(depthmap), 'original')
yaml_file.release()

# resize everything
depthmap = cv2.resize(depthmap, (shape_input[1], shape_input[0]), interpolation=cv2.INTER_LINEAR)
print('resized min max:', np.amin(depthmap), np.amax(depthmap))
show(depthmap, np.amin(depthmap), np.amax(depthmap), 'resized')

depthmap = (max_depth) / np.clip(depthmap, min_depth, max_depth)
show(depthmap, np.amin(depthmap), np.amax(depthmap),
     str(min_depth) + ', ' + str(max_depth) + ', ' + str(np.amin(depthmap)) + ', ' + str(np.amax(depthmap)))
print(np.amin(depthmap), np.amax(depthmap))
