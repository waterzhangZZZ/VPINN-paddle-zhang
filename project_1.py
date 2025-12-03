"""
对流扩散方程的可视化脚本
即使在没有Paddle的环境中也能运行，用于生成易于理解的图片。

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 参数
epsilon = 0.1          # 扩散系数
bx, by = 1.0, 0.0      # 对流速度
x_l, x_r = -1.0, 1.0
y_l, y_r = -1.0, 1.0

# 解析解
def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# 源项 f (根据方程推导)
def f_source(x, y):
    term1 = 2 * epsilon * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)
    term2 = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return term1 + term2

# 生成网格
n = 100
x = np.linspace(x_l, x_r, n)
y = np.linspace(y_l, y_r, n)
xx, yy = np.meshgrid(x, y)

# 计算解析解和源项
u_true = u_exact(xx, yy)
f = f_source(xx, yy)

# 模拟预测解（假设有微小误差）
# 为了演示，我们给解析解添加一些随机扰动作为“预测”
np.random.seed(42)
noise = 0.05 * np.random.randn(*xx.shape) * (1 - xx**2) * (1 - yy**2)  # 在边界处趋于零
u_pred = u_true + noise

# 误差
error = np.abs(u_pred - u_true)

# 创建图形
fig = plt.figure(figsize=(18, 12))

# 1. 解析解的热图
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.pcolormesh(xx, yy, u_true, shading='auto', cmap='viridis')
plt.colorbar(im1, ax=ax1, label='u(x,y)')
ax1.set_title('解析解 $u_{exact}(x,y) = \sin(\pi x) \sin(\pi y)$', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')

# 2. 预测解的热图
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.pcolormesh(xx, yy, u_pred, shading='auto', cmap='viridis')
plt.colorbar(im2, ax=ax2, label='u(x,y)')
ax2.set_title('模拟的VPINN预测解（带有噪声）', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')

# 3. 误差分布
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.pcolormesh(xx, yy, error, shading='auto', cmap='hot_r')
plt.colorbar(im3, ax=ax3, label='绝对误差')
ax3.set_title('绝对误差分布', fontsize=14)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal')

# 4. 对流速度场示意图
ax4 = plt.subplot(2, 3, 4)
# 绘制箭头表示对流方向
X, Y = np.meshgrid(np.linspace(x_l, x_r, 10), np.linspace(y_l, y_r, 10))
U = bx * np.ones_like(X)
V = by * np.ones_like(Y)
ax4.quiver(X, Y, U, V, color='red', scale=20, width=0.005)
ax4.set_xlim(x_l, x_r)
ax4.set_ylim(y_l, y_r)
ax4.set_title('对流速度场 $\\mathbf{b} = (%.1f, %.1f)$' % (bx, by), fontsize=14)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_aspect('equal')
# 添加扩散系数文本
ax4.text(0.05, 0.95, r'$\epsilon = %.1f$' % epsilon,
         transform=ax4.transAxes, fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. 源项 f(x,y) 的热图
ax5 = plt.subplot(2, 3, 5)
im5 = ax5.pcolormesh(xx, yy, f, shading='auto', cmap='plasma')
plt.colorbar(im5, ax=ax5, label='f(x,y)')
ax5.set_title('源项 $f(x,y) = -\\epsilon \\Delta u + \\mathbf{b} \\cdot \\nabla u$', fontsize=14)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_aspect('equal')

# 6. VPINN方法流程图（示意图）
ax6 = plt.subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
# 绘制简单的流程图
boxes = [
    (1, 9, "PDE: $-\\epsilon \\Delta u + \\mathbf{b} \\cdot \\nabla u = f$"),
    (1, 7, "变分形式: $a(u,v) = (f,v)$"),
    (1, 5, "神经网络 $u_\\theta$ 近似解"),
    (1, 3, "损失函数: $L = L_b + L_{strong} + L_{var}$"),
    (1, 1, "优化器更新参数"),
]
for xb, yb, text in boxes:
    ax6.add_patch(plt.Rectangle((xb, yb-0.5), 8, 0.8, fc='lightblue', ec='black'))
    ax6.text(xb+0.2, yb, text, fontsize=10, va='center')
# 连接箭头
for i in range(len(boxes)-1):
    ax6.arrow(5, boxes[i][1]-0.5, 0, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax6.set_title('VPINN方法流程图', fontsize=14)

plt.suptitle('对流扩散方程 $ -\\epsilon \\Delta u + \\mathbf{b} \\cdot \\nabla u = f $ 的VPINN求解示意图', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('convection_diffusion_visualization.png', dpi=300, bbox_inches='tight')
print("图片已保存为 'convection_diffusion_visualization.png'")
plt.show()

# 另外创建一个3D曲面图
fig3d = plt.figure(figsize=(12, 5))
ax3d1 = fig3d.add_subplot(121, projection='3d')
surf1 = ax3d1.plot_surface(xx, yy, u_true, cmap='viridis', alpha=0.8)
ax3d1.set_title('解析解 3D 视图')
ax3d1.set_xlabel('x')
ax3d1.set_ylabel('y')
ax3d1.set_zlabel('u')

ax3d2 = fig3d.add_subplot(122, projection='3d')
surf2 = ax3d2.plot_surface(xx, yy, u_pred, cmap='viridis', alpha=0.8)
ax3d2.set_title('预测解 3D 视图')
ax3d2.set_xlabel('x')
ax3d2.set_ylabel('y')
ax3d2.set_zlabel('u')

plt.tight_layout()
plt.savefig('convection_diffusion_3d.png', dpi=300)
print("3D图片已保存为 'convection_diffusion_3d.png'")
plt.show()