import numpy as np
import cv2 as cv


class AdaptLms:
    """lms filter"""
    alpha = 0
    y = 0
    x = 0
    w = np.random.randn(2,)
    x0 = 0
    y0 = 0
    mx = 0
    my = 0
    beta = 0.025

    def __init__(self, lr):
        self.miu = lr

    def _set_xy(self):
        self.x = 5
        self.y = 17 * self.x + 10 + np.random.randn()

    def get_w(self):
        for idx in range(10000):
            self._set_xy()
            if idx < 40:
                self.mx += self.beta*self.x
                self.my += self.beta*self.y
                x = self.x
                y = self.y
            else:
                self.mx = self.beta*self.x + (1 - self.beta)*self.mx
                self.my = self.beta*self.y + (1 - self.beta)*self.my
                x = self.mx
                y = self.my
            xn = np.array([x, 1])
            en = y - np.dot(self.w, xn)
            self.w = self.w + self.miu * en * xn
            if idx % 100 == 0:
                print(idx, ',', self.w, ',', self.mx, ',', self.my)
            # if idx < 50:


if __name__ == '__main__':
    lms0 = AdaptLms(1e-3)
    lms0.get_w()
    print('done')
