import numpy as np

# 时间步长
td = 1
# 权重因子
p1 = 1  # 舒适性权重
p2 = 1  # 时间权重
p3 = 1e-7  # 能耗权重

# 列车参数 (根据表1.1)
G = 194.295  # 列车运行重量 (t) AW0
g = 10  # 重力加速度 (m/s**2)

# 坡度参数 (根据表1.2)
i1 = -0.9  # 0-161m
i2 = -0.6  # 161-361m
i3 = 3.3   # 361-691m
i4 = 1.2   # 691-1351m
i5 = -2.1  # 1351-1881m


class train_env_ato:
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
        self.total_distance = 1881  # 总运行距离 (m)

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

        # 最大步数限制
        if self.count < 5000:
            # 根据位置更新速度和位置
            last_v = v
            v += action

            # 速度限制 (最大运行速度 80 km/h = 22.22 m/s)
            v_max = 22.22  # 80 km/h 转换为 m/s
            v = np.clip(v, 0, v_max)

            v_mean = (last_v + v) / 2.0
            x += v_mean * td

            # 检查是否到达终点
            if x >= self.total_distance:
                self.done = True
                self.final = 1000
                x = self.total_distance

            # 计算基本阻力 w0 = 2.031 + 0.0622v + 0.001807v² (N/kN)
            # 注意：v的单位是m/s，需要转换为km/h
            v_kmh = v * 3.6  # m/s 转换为 km/h
            w0 = 2.031 + 0.0622 * v_kmh + 0.001807 * (v_kmh ** 2)

            # 根据位置确定坡度
            if x <= 161:
                w = w0 + i1
            elif 161 < x <= 361:
                w = w0 + i2
            elif 361 < x <= 691:
                w = w0 + i3
            elif 691 < x <= 1351:
                w = w0 + i4
            elif 1351 < x <= 1881:
                w = w0 + i5
            else:
                w = w0  # 超出范围

            # 计算功率 (kW)
            # f = w * G * g (N)
            # power = f * v (W) = f * v / 1000 (kW)
            f = w * G * g  # N
            self.P += f * v * td / 1000  # kW

            # 计算奖励
            # r1_comfort: 舒适性奖励（惩罚加速度变化）
            action_ratio = np.abs(action - self.last_action)
            r1_comfort = -action_ratio / td * p1

            # r2_time_lim: 时间奖励（惩罚时间步数）
            r2_time_lim = -1 * p2

            # r3_power: 能耗奖励（惩罚能耗）
            r3_power = -(self.P * p3) * (self.done or self.truncated)

            # 总奖励
            self.r = r1_comfort + r2_time_lim + r3_power + self.final
            self.state = np.array([x, v]).reshape(1, -1).squeeze(0)
            self.last_action = action

        else:
            self.truncated = True
            self.info = 'timeout'

        return self.state, self.r, self.done, self.truncated, self.info