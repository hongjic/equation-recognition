from skimage import io
from scipy import ndimage


# img = io.imread("/Users/chenhongji/Projects/img.png", as_grey=True)
# filtered = ndimage.median_filter(img, 5)
# io.imsave("/Users/chenhongji/Projects/filtered.png", filtered)


# accept img in ndarray format.
def reduce_noise(img):
    return ndimage.median_filter(img, 5)


def two_value(img):
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i][j] > 128:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img


def image_process(img):
    return reduce_noise(two_value(img))


