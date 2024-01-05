import numpy as np
import matplotlib.pyplot as plt

# 设置Henon模型参数
a = 1.4
b = 0.3

# 设置噪声参数
mean = 0  # 均值
std = 0.1  # 标准差

# 设置模型初始状态变量
x = 0.1
y = 0.1

# 设置模拟参数
n = 10000  # 模拟次数
dt = 0.01  # 时间步长

# 初始化列表
x_list = [x]
y_list = [y]
noise_list = [np.random.normal(mean, std)]

# 使用Euler法模拟Henon模型并添加高斯噪声修正项
for i in range(n):
    x_next = 1 - a * x ** 2 + y + noise_list[i]
    y_next = b * x
    x = x_next
    y = y_next
    noise = np.random.normal(mean, std)
    x_list.append(x)
    y_list.append(y)
    noise_list.append(noise)

# 绘制模拟结果
plt.figure(figsize=(10, 6))
plt.plot(x_list, y_list, color='blue', linewidth=0.5)
plt.title('Henon Model with Gaussian Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.show()