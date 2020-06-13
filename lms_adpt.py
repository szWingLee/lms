import numpy as np
import cv2 as cv


class LinearRegression:
    """lms filter"""
    alpha = 0
    y = 0
    x = 0
    w = np.zeros(2, )
    x0 = 0
    y0 = 0
    mx = 0
    my = 0
    mxx = 0
    mxy = 0
    k = 0
    b = 0
    beta = 0.025
    init_cnt = 40
    b0 = 14
    k0 = 2
    x0 = 5
    sum_e = 0
    e_vec: list = []
    sub_offset: bool = False
    err_cnt = 0
    k_err_cnt = 0
    _idx = 0

    def __init__(self, lr, sub_of: bool, init_frame=40):
        self.miu = lr
        self.init_cnt = init_frame
        self.sub_offset = sub_of

    def _set_idx(self):
        self._idx += 1

    def _set_xy(self):
        # self.x = 5 + np.random.randn() * 1
        # self.y = 17 * self.x + 10
        self.x = self.x0 + np.random.uniform(-0.1, 0.1)
        self.y = self.k0 * self.x + self.b0 + np.random.randn() * 0.2

    def get_w(self):
        for idx in range(100000):
            self._set_xy()
            if idx < 40:
                self.mx += self.beta * self.x
                self.my += self.beta * self.y
                x = self.x
                y = self.y
            else:
                self.mx = self.beta * self.x + (1 - self.beta) * self.mx
                self.my = self.beta * self.y + (1 - self.beta) * self.my
                x = self.x
                y = self.y
            xn = np.array([x, 1])
            en = y - np.dot(self.w, xn)
            self.w = self.w + self.miu * en * xn
            if idx % 1000 == 0:
                print(idx, ',', self.w, ',', self.mx, ',', self.my)
            # if idx < 50:

    def _init_xy(self, idx):
        if idx < self.init_cnt:
            self.mx += self.beta * self.x
            self.my += self.beta * self.y
        else:
            self.mx = self.beta * self.x + (1 - self.beta) * self.mx
            self.my = self.beta * self.y + (1 - self.beta) * self.my

    def _init_xy_m(self, idx):
        if idx < self.init_cnt:
            self.my += self.beta * self.y
        else:
            self.my = self.beta * self.y + (1 - self.beta) * self.my

        if self.sub_offset:
            self.mx = self.my - self.b0
        else:
            self.mx = self.my
        self.x = self.mx

    def _init_cov(self, idx):
        if idx < self.init_cnt:
            # self.mxx += self.beta * self.x * self.x
            # self.mxy += self.beta * self.x * self.y
            return
        else:
            self.mxx = self.beta * self.x * self.x + (1 - self.beta) * self.mxx
            self.mxy = self.beta * self.y * self.x + (1 - self.beta) * self.mxy

    def lms(self, batch):
        self._init_xy()
        x = self.x
        y = self.y
        xn = np.array([x, 1])
        en = y - np.dot(self.w, xn)
        self.w = self.w + self.miu * en * xn
        if self._idx % batch == 0:
            print(self._idx, ',', self.w, ',', self.mx, ',', self.my)
        # if idx < 50:

    def calc_lms(self, loop, batch):
        for idx in range(loop):
            self._set_idx()
            self._set_xy()
            self.lms(idx, batch)

    def lr_online(self, batch):
        self._init_xy(self._idx)
        self._init_cov(self._idx)
        self.k = (self.mxy - self.mx * self.my) / (self.mxx - self.mx * self.mx)
        self.b = self.my - self.k * self.mx
        if self._idx % batch == 0:
            print(self._idx, '', self.k, ',', self.b)
        if self._idx > self.init_cnt:
            self._rec_err()
        return

    def calc_lr(self, loop, batch):
        for idx in range(loop):
            self._set_idx()
            self._set_xy()
            self.lr_online(idx, batch)

    def _rec_err(self):
        x = (self.y - self.b) / self.k
        self.e_vec.append(x)
        mm = self.mx
        dif = abs(x - mm) / mm
        if dif > 0.5:
            self.err_cnt += 1
            # print('error too big')

    def lr_online_m(self, batch):
        self._init_xy_m(self._idx)
        self._init_cov(self._idx)
        if self._idx >= self.init_cnt:
            rxy = (self.mxy - self.mx * self.my)
            rxx = (self.mxx - self.mx * self.mx)
            k = rxy / rxx
            if (self.k > 0) and (((abs(k - self.k) / self.k) > 0.5) or (k < 1e-4)):
                self.k_err_cnt += 1
            else:
                self.k = k
            self.b = self.my - self.k * self.mx
            self._rec_err()
            if self._idx % batch == 0:
                print(self._idx)
        return

    def calc_lr_m(self, loop, batch):
        for idx in range(loop):
            self._set_idx()
            self._set_xy()
            self.lr_online_m(batch)


if __name__ == '__main__':
    see = 0
    cnt = 0
    sub_base = True
    err_cnt = 0
    k_err_cnt = 0
    in_loop = int(1e4)
    out_loop = int(100)
    in_batch = in_loop
    all_loop = (in_loop * out_loop)
    for ix in range(out_loop):
        lr0 = LinearRegression(0.05, sub_base)
        # lms0.get_w()
        # lr0.calc_lms(int(1e5), int(1e3))
        # lr0.calc_lr(int(1e4), int(1e4))
        lr0.calc_lr_m(in_loop, in_batch)
        ee = np.std(lr0.e_vec) / np.mean(lr0.e_vec)
        if ee > 0.5:
            cnt += 1
            print('wrong ', lr0.err_cnt)
        err_cnt += lr0.err_cnt
        k_err_cnt += lr0.k_err_cnt
        print('x-wrong ', lr0.err_cnt)
        see += ee
    print('done', see / 100, 'e-x:', err_cnt / all_loop, 'e-k:', k_err_cnt / all_loop)
