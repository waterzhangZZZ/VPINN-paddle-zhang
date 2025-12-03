import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

d_type = "float32"
np.random.seed(1234)
paddle.seed(seed=1234)

class Sin(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.sin(x)

class PINN_CD(paddle.nn.Layer):
    def __init__(
        self,
        X_bc,           # 边界点坐标，shape=[num_bc, 2]
        u_bc,           # 边界点真实解，shape=[num_bc, 1]
        layers,
        f_ext,          # 源项函数
        epsilon=0.1,    # 扩散系数
        b=(1.0, 0.0)    # 对流速度 (bx, by)
    ):
        super(PINN_CD, self).__init__()
        self.f_ext = f_ext

        # 定义网络结构
        paddle.set_default_dtype(d_type)
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(Sin())
        self.net = nn.Sequential(*net_layers)
        self.LR = 0.001
        self.optimizer_Adam = paddle.optimizer.Adam(
            parameters=self.parameters(), learning_rate=self.LR)

        # 边界数据
        self.X_bc = paddle.to_tensor(X_bc, dtype=d_type, stop_gradient=False)
        self.u_bc = paddle.to_tensor(u_bc, dtype=d_type, stop_gradient=False).reshape([-1, 1])

        # 扩散系数和对流速度
        self.epsilon = epsilon
        self.bx = b[0]
        self.by = b[1]

        # 内部训练点（用于强形式损失）与VPINN相同数量
        n_train_point = 1000
        self.X_train = paddle.uniform(shape=[n_train_point, 2], min=-1, max=1, dtype=d_type)
        self.X_train.stop_gradient = False

    def forward(self, x, y):
        """前向传播"""
        if isinstance(x, np.ndarray):
            with paddle.no_grad():
                x = paddle.to_tensor(x, dtype=d_type)
        if isinstance(y, np.ndarray):
            with paddle.no_grad():
                y = paddle.to_tensor(y, dtype=d_type)

        u = self.net_u(x, y)
        return u

    def net_u(self, x, y):
        # 确保x和y是二维张量
        if len(x.shape) == 1:
            x = x.reshape([-1, 1])
        if len(y.shape) == 1:
            y = y.reshape([-1, 1])
        return self.net(paddle.concat([x, y], axis=1))

    def net_du(self, x, y):
        """计算u对x和y的一阶、二阶导数"""
        x.stop_gradient = False
        y.stop_gradient = False
        u = self.forward(x, y)

        # 一阶导数
        d1xu = paddle.grad(u, x, grad_outputs=paddle.ones_like(u), create_graph=True)[0]
        d1yu = paddle.grad(u, y, grad_outputs=paddle.ones_like(u), create_graph=True)[0]

        # 二阶导数
        d2xu = paddle.grad(d1xu, x, grad_outputs=paddle.ones_like(d1xu), create_graph=True)[0]
        d2yu = paddle.grad(d1yu, y, grad_outputs=paddle.ones_like(d1yu), create_graph=True)[0]

        return d1xu, d1yu, d2xu, d2yu

    def loss(self):
        # 边界损失
        x_bc = self.X_bc[:, 0].reshape([-1, 1])
        y_bc = self.X_bc[:, 1].reshape([-1, 1])
        u_bc_pred = self.net_u(x_bc, y_bc)
        lossb = paddle.mean((u_bc_pred - self.u_bc) ** 2)

        # 强形式损失（内部点）
        x_train = self.X_train[:, 0].reshape([-1, 1])
        y_train = self.X_train[:, 1].reshape([-1, 1])
        d1xu, d1yu, d2xu, d2yu = self.net_du(x_train, y_train)
        f_train = self.f_ext(x_train, y_train)
        # 强形式残差： -epsilon*(d2xu + d2yu) + bx*d1xu + by*d1yu - f_train = 0
        residual = -self.epsilon * (d2xu + d2yu) + self.bx * d1xu + self.by * d1yu - f_train
        lossad = paddle.mean(residual ** 2)

        # 总损失
        lossb_weight = 100
        loss = lossb_weight * lossb + lossad
        return loss, lossb, lossad

    def train(self, N_iter, tresh):
        """
        训练函数
        """
        start_time = time.time()
        for it in range(N_iter):
            self.optimizer_Adam.clear_grad()
            loss, lossb, lossad = self.loss()
            if (it + 1) % 1000 == 0:
                time_now = time.time()
                time_cost = time_now - start_time
                start_time = time_now
                print("Iter: %d, Loss: %.3e, Lossb: %.3e, Lossad: %.3e" % (it+1, loss.numpy(), lossb.numpy(), lossad.numpy()))
                print("Time cost: %.2f" % (time_cost))
            if loss < tresh:
                break

            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()


if __name__ == "__main__":
    """
    参数定义（与VPINN保持一致）
    """
    LR = 0.001
    N_iter = 5000
    tresh = 2e-32
    Net_layer = [2] + [20] * 3 + [1]
    lossb_weight = 100
    x_l, x_r = -1.0, 1.0
    y_l, y_r = -1.0, 1.0
    epsilon = 0.1          # 扩散系数
    b = (1.0, 0.0)         # 对流速度 (bx, by)

    """
    u_ext:方程的解析解
    f_ext:方程的外部力定义（根据解析解推导）
    """
    def u_ext(x, y):
        u = paddle.sin(np.pi * x) * paddle.sin(np.pi * y)
        return u

    def f_ext(x, y):
        term1 = 2 * epsilon * (np.pi ** 2) * paddle.sin(np.pi * x) * paddle.sin(np.pi * y)
        term2 = np.pi * paddle.cos(np.pi * x) * paddle.sin(np.pi * y)
        return term1 + term2

    """
    边界点：方形区域四条边上的点
    """
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

    # 边界上的真实解
    u_bc = u_ext(paddle.to_tensor(X_bc[:, 0]), paddle.to_tensor(X_bc[:, 1])).numpy()

    # 创建模型
    model = PINN_CD(
        X_bc=X_bc,
        u_bc=u_bc,
        layers=Net_layer,
        f_ext=f_ext,
        epsilon=epsilon,
        b=b,
    )

    # 训练
    model.train(N_iter, tresh)

    # 评估
    n_plot = 100
    x_plot = np.linspace(x_l, x_r, n_plot)
    y_plot = np.linspace(y_l, y_r, n_plot)
    xx, yy = np.meshgrid(x_plot, y_plot)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))

    with paddle.no_grad():
        u_pred = model(
            paddle.to_tensor(grid_points[:, 0], dtype=d_type).reshape([-1, 1]),
            paddle.to_tensor(grid_points[:, 1], dtype=d_type).reshape([-1, 1])
        )
        u_pred = u_pred.numpy().reshape(n_plot, n_plot)

    # 真实解
    u_true = np.sin(np.pi * xx) * np.sin(np.pi * yy)

    # 误差
    error = np.abs(u_pred - u_true)
    max_error = np.max(error)
    avg_error = np.mean(error)
    print(f"Maximum absolute error: {max_error:.4e}")
    print(f"Average absolute error: {avg_error:.4e}")

    # 绘图
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.pcolormesh(xx, yy, u_pred, shading='auto', cmap='viridis')
    plt.colorbar(label='Predicted Value')
    plt.title('PINN Prediction')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 3, 2)
    plt.pcolormesh(xx, yy, u_true, shading='auto', cmap='viridis')
    plt.colorbar(label='True Value')
    plt.title('Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 3, 3)
    error_plot = plt.pcolormesh(xx, yy, error, shading='auto', cmap='hot_r')
    plt.colorbar(error_plot, label='Absolute Error')
    plt.title('Error Distribution')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig('pinn_convection_diffusion_results.png', dpi=300)
    plt.show()