import os.path
from glob import glob
from natsort import os_sorted
import cv2
import matplotlib.pyplot as plt


def compare_imgs(img1_path, img2_path):
    for i, (im1, im2) in enumerate(zip(img1_path, img2_path)):
        img1 = cv2.imread(im1)[:, :, ::-1]
        img2 = cv2.imread(im2)[:, :, ::-1]

        plt.subplot(121)
        plt.imshow(img1)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(img2)
        plt.axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(rf'E:\Dataset\SR\Benchmark\Manga109\test\{i}.png')

if __name__ == '__main__':
    img1_path = os_sorted(glob(r'E:\Dataset\SR\Benchmark\decoder\*'))
    img2_path = os_sorted(glob(r'E:\Dataset\SR\Benchmark\Manga109\LRbicx2\*'))

    compare_imgs(img1_path, img2_path)