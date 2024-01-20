import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
动态绘制目标数目变化折线图
"""
class TargetPlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xs = []
        self.ys = []

    def update(self, target_num):
        self.xs.append(len(self.xs))
        self.ys.append(target_num)
        self.ax.clear()
        self.ax.plot(self.xs, self.ys)
        plt.title('Dynamic Target Number Line Chart')
        plt.xlabel('Frame')
        plt.ylabel('Target Number')

    def show(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=1000)
        plt.show()