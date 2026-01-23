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
max_acceleration = 1  # 最大加速度 (m/s²)
max_deceleration = 1  # 最大减速度 (m/s²)
v_max_global = 22.22  # 最大限速 80 km/h = 22.22 m/s

# 坡度参数 (根据表1.2)
i1 = -0.9  # 0-161m
i2 = -0.6  # 161-361m
i3 = 3.3   # 361-691m
i4 = 1.2   # 691-1351m
i5 = -2.1  # 1351-1881m
i6 = -0.7  # 1881-2251m
i7 = 1.1   # 2251-2451m
i8 = -0.6  # 2451-2632m


class train_env_ato:
    def __init__(self):
        self.state = np.zeros(2, dtype=np.float32)
        self.r = 0
        self.done = False
        self.truncated = False
        self.info = 'successful'
        self.last_action = 0
        self.count = 0
        self.final = 0
        self.P = 0
        self.total_distance = 2632  # 总运行距离 (m)

    def reset(self):
        self.state = np.zeros(2, dtype=np.float32)
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
            # 限制加速度范围
            action = np.clip(action, -max_deceleration, max_acceleration)

            # 根据位置更新速度和位置
            last_v = v
            v += action

            # 速度限制：使用全局最大限速 80 km/h
            v = np.clip(v, 0, v_max_global)

            v_mean = (last_v + v) / 2.0
            x += v_mean * td

            # 检查是否到达终点
            if x >= self.total_distance:
                self.done = True
                # 基础完成奖励：只要完成就给固定奖励
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
            elif 1881 < x <= 2251:
                w = w0 + i6
            elif 2251 < x <= 2451:
                w = w0 + i7
            elif 2451 < x <= 2632:
                w = w0 + i8
            else:
                w = w0  # 超出范围

            # 计算功率
            f = w * G * g + G * 1000 * action  # 总牵引力（N）
            self.P += f * v * td  # 功率 = 力 × 速度 × 时间

            # 计算奖励
            action_ratio = np.abs(action - self.last_action)
            r1_comfort = -action_ratio / td * p1  # 舒适度惩罚
            r2_time_lim = -1 * p2  # 时间惩罚
            r3_power = -(self.P * p3) * (self.done or self.truncated)  # 能耗惩罚
            self.r = r1_comfort + r2_time_lim + r3_power + self.final
            # 确保 x 和 v 是标量
            x_scalar = float(x) if hasattr(x, '__iter__') else x
            v_scalar = float(v) if hasattr(v, '__iter__') else v
            self.state = np.array([x_scalar, v_scalar], dtype=np.float32)
            self.last_action = action

        else:
            self.truncated = True
            self.info = 'timeout'

        return self.state, self.r, self.done, self.truncated, self.info