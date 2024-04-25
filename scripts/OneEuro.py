import numpy as np
import math


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, x0, dx0=0.0, min_cutoff=0.001, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = np.full(x0.shape, min_cutoff).astype(np.float32)
        self.beta = np.full(x0.shape, beta).astype(np.float32)
        self.d_cutoff = np.full(x0.shape, d_cutoff).astype(np.float32)
        # Previous values.
        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.full(x0.shape, dx0).astype(np.float32)

    def __call__(self, x):
        """Compute the filtered signal."""
        t_e = 0.011
        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class EuroPose:
    def __init__(self):
        self.key_point = None
        self.pre_key_point = None
        self.one_euro_filter = None
        self.num_frame = 1

    def one_euro_pose(self, input_key_point, result):

        if self.num_frame == 1:

            self.key_point = input_key_point

            if result[0].size == 0:  # 当输入进来的点没有任何异常才可以进入
                min_cutoff = 0.0001
                beta = 0.1
                self.one_euro_filter = OneEuroFilter(
                    input_key_point,
                    min_cutoff=min_cutoff,
                    beta=beta
                )
                self.num_frame += 1
                self.key_point = input_key_point
                self.pre_key_point = input_key_point
        else:
            if result[0].size > 0:
                for i in range(len(result[0])):
                    self.key_point[result[0][i]] = self.pre_key_point[result[0][i]]
            self.key_point = self.one_euro_filter(input_key_point)
            self.pre_key_point = self.key_point
