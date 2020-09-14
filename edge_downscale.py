import cv2 as cv
import numpy as np


class DownScale:

    def __init__(self, scale, src, sobel_size):
        self.scale = scale
        self.src: np.ndarray = src
        self.sobel_size = sobel_size
        self.edge: np.ndarray = np.zeros(src.shape)
        self.gray = cv.cvtColor(src[0:1440, 0:1440, :], cv.COLOR_BGR2GRAY)
        self.rows, self.cols = self.gray.shape
        self.down_rows = self.rows // self.scale
        self.down_cols = self.cols // self.scale
        self.resize_image: np.ndarray = np.zeros((self.down_rows, self.down_cols))

    def edge_detect(self):
        gx = cv.Sobel(self.gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=self.sobel_size)
        gy = cv.Sobel(self.gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=self.sobel_size)
        gradient = gx * gx + gy * gy
        grd_m = np.mean(gradient)
        self.edge = (gradient > grd_m) * 255
        return self.edge

    def down_blob(self, x, y):
        k = self.scale // 2
        offset = (k % 2 == 0)
        values = []
        # mask = []
        for x_ in range(-k, (k + offset), 1):
            for y_ in range(-k, (k + offset), 1):
                values.append(self.gray[x_ + x, y_ + y])
                # mask.append(self.edge[x_ + x, y_ + y])
        mean_val = np.mean(values)
        if self.edge[x, y] == 0:
            downscale_value = mean_val
        else:
            # downscale_value = self.gray[x, y]
            sort_val = np.sort(values, axis=None)
            median_val = np.median(values)
            if mean_val < median_val:
                # downscale_value = median_val
                # downscale_value = np.mean(values[0:2])
                downscale_value = np.mean(values[-self.scale::])
            else:
                downscale_value = np.mean(values[0:self.scale])
                # downscale_value = np.mean(values[-2::])
        return downscale_value

    def img_resize(self):
        # bmp = cv.resize(self.gray, (self.rows//self.scale, self.cols//self.scale), interpolation=cv.INTER_BITS)
        bmp = cv.resize(self.gray, (560, 560), interpolation=cv.INTER_BITS)
        return bmp

    def img_direct_sample(self):
        k = self.scale // 2
        dst = self.gray[k::self.scale, k::self.scale]
        return dst

    def down_image(self):
        offset = self.scale // 2
        for x_ in range(self.down_rows):
            x = offset + x_ * self.scale
            for y_ in range(self.down_cols):
                y = offset + y_ * self.scale
                self.resize_image[x_, y_] = self.down_blob(x, y)
        self.resize_image = np.round(self.resize_image)
        return self.resize_image.astype(np.uint8)


if __name__ == '__main__':
    img = cv.imread('timg.jpg')
    img_resize = DownScale(scale=3, src=img, sobel_size=7)
    # cv.imwrite('gray.png', img_resize.gray)
    img_edge = img_resize.edge_detect()
    # cv.imwrite('edge-3.png', img_edge)
    img_scaled = img_resize.down_image()
    cv.imwrite('downscale-mk.png', img_scaled)
    # cv.imwrite('resize-d.png', img_resize.img_direct_sample())
    # cv.imwrite('resize-bits.png', img_resize.img_resize())
    print('done')
