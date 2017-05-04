from glob import glob
import re
import shutil
import shelve


def separate_images():
    all_image_paths = glob("annotated/*.png")
    labels = shelve.open("labels.shelve")
    for path in all_image_paths:
        s_path = path[10:]
        path_arr = s_path.split("_")
        equation = path_arr[2]
        if len(path_arr) > 3:
            shutil.copyfile(path, "symbols/" + s_path)
            labels[s_path] = path_arr[3]
        else:
            shutil.copyfile(path, "equations/" + s_path)
    labels.close()


if __name__ == "__main__":
    separate_images()