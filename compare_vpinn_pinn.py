import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights

# 设置随机种子以确保可重复性
d_type = "float32"
np.random.seed(1234)
paddle.seed(seed=1234)

# 公共参数
LR = 0.001
N_iter = 5000               # 训练迭代次数
tresh = 2e-32               # 提前停止阈值（实际上很难达到）
Net_layer = [2] + [20] * 3 + [1]   # 网络结构
lossb_weight = 100          # 边界损失权重
lossv_weight = 1            # 变分损失权重（仅VPINN使用）
x_l, x_r = -1.0, 1.0
y_l, y_r = -1.0, 1.0
epsilon = 0.1               # 扩散系数
b = (1.0, 0.0)              # 对流速度 (bx, by)
N_testfcn = 40              # 测试函数个数（VPINN）
N_quad = 100                # 每个方向的积分点数量（VPINN）

# 解析解和源项
def u_ext(x, y):
    return paddle.sin(np.pi * x) * paddle.sin(np.pi * y)

def f_ext(x, y):
    term1 = 2 * epsilon * (np.pi ** 2) * paddle.sin(np.pi * x) * paddle.sin(np.pi * y)
    term2 = np.pi * paddle.cos(np.pi * x) * paddle.sin(np.pi * y)
    return term1 + term2

# 生成边界点（与两个模型相同）
n_bc = 100
bc_down_x = np.linspace(x_l, x_r, n_bc)
bc_down_y = np.full(n_bc, y_l)
bc_up_x = np.linspace(x_l, x_r, n_bc)
bc_up_y = np.full(n_bc, y_r)
bc_left_x = np.full(n_bc, x_l)
bc_left_y = np.linspace(y_l, y_r, n_bc)
bc_right_x = np.full(n_bc, x_r)
bc_right_y = np.linspace(y_l, y_r, n_bc)

X_bc = np.vstack([
    np.column_stack((bc_down_x, bc_down_y)),
    np.column_stack((bc_up_x, bc_up_y)),
    np.column_stack((bc_left_x, bc_left_y)),
    np.column_stack((bc_right_x, bc_right_y))
])
# 边界真实解
u_bc = u_ext(paddle.to_tensor(X_bc[:, 0]), paddle.to_tensor(X_bc[:, 1])).numpy()

# 生成积分点（供VPINN使用）
[X_quad_1d, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
Y_quad_1d, WY_quad = X_quad_1d, WX_quad
xx, yy = np.meshgrid(X_quad_1d, Y_quad_1d)
wxx, wyy = np.meshgrid(WX_quad, WY_quad)
XY_quad = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
WXY_quad = (wxx * wyy).flatten()[:, None]

# 导入模型类
# 由于文件名包含连字符，不能直接import，使用importlib
import importlib
vpinn_module = importlib.import_module("VPINN-ConvectionDiffusion")
VPINN_CD = vpinn_module.VPINN_CD
from PINN_ConvectionDiffusion import PINN_CD

# 记录训练过程的数据结构
class Trainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.loss_history = []
        self.lossb_history = []
        self.lossad_history = []
        self.lossv_history = []  # 仅VPINN使用
        self.times = []
        self.start_time = None

    def train(self, N_iter, tresh):
        self.start_time = time.time()
        for it in range(N_iter):
            self.model.optimizer_Adam.clear_grad()
            if self.model_name == 'VPINN':
                loss, lossb, lossad, lossv = self.model.loss()
                self.lossv_history.append(lossv.numpy())
            else:
                loss, lossb, lossad = self.model.loss()
                lossv = None
            loss.backward(retain_graph=True)
            self.model.optimizer_Adam.step()

            # 记录
            self.loss_history.append(loss.numpy())
            self.lossb_history.append(lossb.numpy())
            self.lossad_history.append(lossad.numpy())
            if lossv is not None:
                self.lossv_history.append(lossv.numpy())
            self.times.append(time.time() - self.start_time)

            # 提前停止
            if loss.numpy() < tresh:
                print(f"{self.model_name} 在迭代 {it+1} 提前停止")
                break

            if (it + 1) % 1000 == 0:
                print(f"{self.model_name} Iter {it+1}: Loss = {loss.numpy():.3e}, Lossb = {lossb.numpy():.3e}, Lossad = {lossad.numpy():.3e}")
        print(f"{self.model_name} 训练完成，总时间 {self.times[-1]:.2f} 秒")

    def evaluate(self):
        n_plot = 100
        x_plot = np.linspace(x_l, x_r, n_plot)
        y_plot = np.linspace(y_l, y_r, n_plot)
        xx, yy = np.meshgrid(x_plot, y_plot)
        grid_points = np.column_stack((xx.flatten(), yy.flatten()))
        with paddle.no_grad():
            u_pred = self.model(
                paddle.to_tensor(grid_points[:, 0], dtype=d_type).reshape([-1, 1]),
                paddle.to_tensor(grid_points[:, 1], dtype=d_type).reshape([-1, 1])
            )
            u_pred = u_pred.numpy().reshape(n_plot, n_plot)
        u_true = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        error = np.abs(u_pred - u_true)
        max_error = np.max(error)
        avg_error = np.mean(error)
        return u_pred, u_true, error, max_error, avg_error

# 创建VPINN模型并训练
print("="*50)
print("开始训练 VPINN")
print("="*50)
vpinn_model = VPINN_CD(
    X_bc=X_bc,
    u_bc=u_bc,
    X_quad=XY_quad,
    w_quad=WXY_quad,
    N_test=N_testfcn,
    layers=Net_layer,
    f_ext=f_ext,
    epsilon=epsilon,
    b=b,
)
# 注意：VPINN_CD类中使用了全局变量LR, lossb_weight, lossv_weight，我们需要设置这些全局变量
# 由于这些变量已经在模块中定义，我们直接传入即可（通过修改类的属性？）
# 但为了简单，我们在导入前设置环境变量？实际上，VPINN_CD类内部使用了来自其模块的全局变量LR等。
# 我们已经在代码开头定义了这些变量，因此导入的类会使用这些值。
vpinn_trainer = Trainer(vpinn_model, 'VPINN')
vpinn_trainer.train(N_iter, tresh)
vpinn_u_pred, vpinn_u_true, vpinn_error, vpinn_max_err, vpinn_avg_err = vpinn_trainer.evaluate()

# 创建PINN模型并训练
print("="*50)
print("开始训练 PINN")
print("="*50)
pinn_model = PINN_CD(
    X_bc=X_bc,
    u_bc=u_bc,
    layers=Net_layer,
    f_ext=f_ext,
    epsilon=epsilon,
    b=b,
)
pinn_trainer = Trainer(pinn_model, 'PINN')
pinn_trainer.train(N_iter, tresh)
pinn_u_pred, pinn_u_true, pinn_error, pinn_max_err, pinn_avg_err = pinn_trainer.evaluate()

# 输出结果对比
print("\n" + "="*50)
print("结果对比")
print("="*50)
print(f"VPINN 最大绝对误差: {vpinn_max_err:.4e}, 平均绝对误差: {vpinn_avg_err:.4e}")
print(f"PINN  最大绝对误差: {pinn_max_err:.4e}, 平均绝对误差: {pinn_avg_err:.4e}")
print(f"VPINN 训练时间: {vpinn_trainer.times[-1]:.2f} 秒")
print(f"PINN  训练时间: {pinn_trainer.times[-1]:.2f} 秒")

# 绘制对比图
fig = plt.figure(figsize=(20, 12))

# 1. 损失曲线对比
ax1 = plt.subplot(2, 3, 1)
ax1.plot(vpinn_trainer.loss_history, label='VPINN Total Loss', color='blue', linewidth=2)
ax1.plot(pinn_trainer.loss_history, label='PINN Total Loss', color='red', linewidth=2, linestyle='--')
ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('Total Loss Comparison')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(vpinn_trainer.lossb_history, label='VPINN Boundary Loss', color='green', linewidth=2)
ax2.plot(pinn_trainer.lossb_history, label='PINN Boundary Loss', color='orange', linewidth=2, linestyle='--')
ax2.set_yscale('log')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Boundary Loss')
ax2.set_title('Boundary Loss Comparison')
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(vpinn_trainer.lossad_history, label='VPINN Strong Loss', color='purple', linewidth=2)
ax3.plot(pinn_trainer.lossad_history, label='PINN Strong Loss', color='brown', linewidth=2, linestyle='--')
ax3.set_yscale('log')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Strong Loss')
ax3.set_title('Strong Form Loss Comparison')
ax3.legend()
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# 2. 变分损失（仅VPINN）
ax4 = plt.subplot(2, 3, 4)
ax4.plot(vpinn_trainer.lossv_history, label='VPINN Variational Loss', color='darkcyan', linewidth=2)
ax4.set_yscale('log')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Variational Loss')
ax4.set_title('VPINN Variational Loss')
ax4.legend()
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

# 3. 误差分布对比
ax5 = plt.subplot(2, 3, 5)
im5 = ax5.pcolormesh(xx, yy, vpinn_error, shading='auto', cmap='hot_r')
plt.colorbar(im5, ax=ax5, label='Absolute Error')
ax5.set_title(f'VPINN Error (Max={vpinn_max_err:.2e})')
ax5.set_xlabel('x')
ax5.set_ylabel('y')

ax6 = plt.subplot(2, 3, 6)
im6 = ax6.pcolormesh(xx, yy, pinn_error, shading='auto', cmap='hot_r')
plt.colorbar(im6, ax=ax6, label='Absolute Error')
ax6.set_title(f'PINN Error (Max={pinn_max_err:.2e})')
ax6.set_xlabel('x')
ax6.set_ylabel('y')

plt.tight_layout()
plt.savefig('comparison_vpinn_pinn.png', dpi=300)
plt.show()

# 保存数据以便后续分析
import pickle
data = {
    'vpinn': {
        'loss': vpinn_trainer.loss_history,
        'lossb': vpinn_trainer.lossb_history,
        'lossad': vpinn_trainer.lossad_history,
        'lossv': vpinn_trainer.lossv_history,
        'time': vpinn_trainer.times,
        'max_error': vpinn_max_err,
        'avg_error': vpinn_avg_err,
    },
    'pinn': {
        'loss': pinn_trainer.loss_history,
        'lossb': pinn_trainer.lossb_history,
        'lossad': pinn_trainer.lossad_history,
        'time': pinn_trainer.times,
        'max_error': pinn_max_err,
        'avg_error': pinn_avg_err,
    }
}
with open('comparison_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("数据已保存到 comparison_data.pkl")