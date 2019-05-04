import os
import matplotlib.pyplot as plt

# directories to which to save the figures
PROJECT_ROOT_DIR = "../"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")


def save_fig(fig_id, tight_layout=False, folder=False, fig_extension="png", resolution=200):
    """
    This function will save figures into the images folder with the following
    defaults unless otherwise specified.

    :param fig_id: name of the file
    :param tight_layout: boolean (default: False)
    :param folder: name of folder (default: False)
    :param fig_extension: file extension (default: png)
    :param resolution: figure resolution (default: 200)
    :return: saved figure in specified folder
    """

    if folder:
        path = os.path.join(IMAGES_PATH, folder, fig_id + "." + fig_extension)
    else:
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print(f'Saving figure: {path}')

    if tight_layout:
        plt.tight_layout()

    plt.savefig(fname=path, format=fig_extension, dpi=resolution)

    return
