import numpy as np

# 时间步长
td = 1
# 权重因子
p1 = 1
p2 = 1
p3 = 1e-7
# 列车参数
G = 120  # t
g = 10  # m/s**2
i1 = 0  # 1/1000
i2 = 4
i3 = 5
i4 = -1


class train_env1:
    def __init__(self):
        self.state = np.zeros([1, 2]).squeeze(0)
        self.r = 0
        self.done = False
        self.truncated = False
        self.info = 'successful'
        self.last_action = 0
        self.count = 0
        self.final = 0
        self.P = 0

    def reset(self):
        self.state = np.zeros([1, 2]).squeeze(0)
        self.r = 0
        self.done = False
        self.truncated = False
        self.info = 'successful'
        self.last_action = 0
        self.count = 0
        self.final = 0
        self.P = 0
        return self.state, self.info

    def step(self, action):
        x, v = self.state
        self.done = False
        self.truncated = False
        self.info = 'successful'
        self.count += 1
        if self.count < 5000:
            if x <= 127:
                last_v = v
                v += action
                v = np.clip(v, 0, 14.4)
                v_mean = (last_v + v) / 2.0
                x += v_mean * td

            elif 127 < x < 1529:
                last_v = v
                v += action
                v = np.clip(v, 0, 20.5)
                v_mean = (last_v + v) / 2.0
                x += v_mean * td

            elif 1529 <= x < 2000:
                last_v = v
                v += action
                v = np.clip(v, 0, 13.3)
                v_mean = (last_v + v) / 2.0
                x += v_mean * td
                x = np.clip(x, 0, 2000)

            elif x >= 2000:
                self.done = True
                self.final = 1000

            w0 = 1.5e-2 + 5e-5 * v + 6e-6 * v ** 2

            if x <= 455:
                w = w0 + i1
                f = w * G * g + G * 1000 * action
                self.P += f * v * td

            elif 455 < x <= 814:
                w = w0 + i2
                f = w * G * g + G * 1000 * action
                self.P += f * v * td

            elif 814 < x <= 1001:
                w = w0 + i3
                f = w * G * g + G * 1000 * action
                self.P += f * v * td

            elif 1001 < x <= 2000:
                w = w0 + i4
                f = w * G * g + G * 1000 * action
                self.P += f * v * td

            action_ratio = np.abs(action - self.last_action)
            r1_comfort = - action_ratio / td * p1
            r2_time_lim = -1 * p2
            r3_power = -(self.P * p3) * (self.done or self.truncated)
            self.r = r1_comfort + r2_time_lim + r3_power + self.final
            self.state = np.array([x, v]).reshape(1, -1).squeeze(0)

        else:
            self.truncated = True

        return self.state, self.r, self.done, self.truncated, self.info
