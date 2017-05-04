import numpy
from skimage import io, measure
from imageutil import imageprocess


# input an image 
# return the image after preprocessing, component boundaries and labelling
def label_components(image):
    rows, cols = image.shape
    image = imageprocess(image)
    cc, number = measure.label(image, background=0, return_num=True, connectivity=2)
    # component boundaries
    cbs = [[rows, cols, 0, 0] for i in range(number)]
    for i in range(rows):
        for j in range(cols):
            label = cc[i][j] - 1
            if label < 0: 
                continue
            cbs[label][0] = min(cbs[label][0], i)
            cbs[label][1] = min(cbs[label][1], j)
            cbs[label][2] = max(cbs[label][2], i)
            cbs[label][3] = max(cbs[label][3], j)
    return (image, cbs, cc)


# output a list of small images
def segment_components(image, cbs, cc):
    components = []
    number = len(cbs)
    for i in range(number):
        label = i + 1
        x1, y1, x2, y2 = cbs[i][0], cbs[i][1], cbs[i][2], cbs[i][3]
        component = numpy.zeros((x2 - x1 + 1, y2 - y1 + 1), dtype=int)
        for row in range(x1, x2 + 1):
            for col in range(y1, y2 + 1):
                if cc[row][col] == label:
                    component[row - x1][col - y1] = image[row][col]
        components.append(component)
    return components


def save_images(components):
    num = len(components)
    for i in range(num):
        io.imsave("images/components/tmp" + str(i + 1) + ".png", components[i])


if __name__ == "__main__":
    image = io.imread("images/equations/SKMBT_36317040717260_eq7.png", as_grey=True)
    image, cbs, label = label_components(image)
    # cut out the components
    components = segment_components(image, cbs, label)
    save_images(components)