import paddle
import paddle.nn as nn
import numpy as np
from scipy.special import eval_legendre
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights, Jacobi, DJacobi
import matplotlib.pyplot as plt
import time

def safe_DJacobi(n, a, b, x, k):
    """
    安全的雅可比多项式导数计算，当 n-k < 0 时返回零。
    """
    if n - k < 0:
        return np.zeros_like(x)
    else:
        return DJacobi(n, a, b, x, k)

d_type = "float32"
np.random.seed(1234)
paddle.seed(seed=1234)


class Sin(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.sin(x)


class VPINN_CD(paddle.nn.Layer):
    def __init__(
        self,
        X_bc,           # 边界点坐标，shape=[num_bc, 2]
        u_bc,           # 边界点真实解，shape=[num_bc, 1]
        X_quad,         # 内部积分点，shape=[num_quad, 2]
        w_quad,         # 积分权重，shape=[num_quad, 1]
        N_test,         # 测试函数个数（每个方向）
        layers,
        f_ext,          # 源项函数
        epsilon=0.1,    # 扩散系数
        b=(1.0, 0.0)    # 对流速度 (bx, by)
    ):
        super(VPINN_CD, self).__init__()
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

        # 积分数据
        self.X_quad = paddle.to_tensor(X_quad, dtype=d_type, stop_gradient=False)
        self.w_quad = paddle.to_tensor(w_quad, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.N_quad = self.X_quad.shape[0]
        self.N_test = N_test

        # 内部训练点（用于强形式损失）
        n_train_point = 1000
        self.X_train = paddle.uniform(shape=[n_train_point, 2], min=-1, max=1, dtype=d_type)
        self.X_train.stop_gradient = False

        # 扩散系数和对流速度
        self.epsilon = epsilon
        self.bx = b[0]
        self.by = b[1]

        # 计算测试函数及其梯度在积分点处的值
        self.testfcn_quad_element = self.Test_fcn(self.N_test, self.X_quad)          # shape=(N_test, N_quad, 1)
        self.dtestfcn_x_quad_element = self.dTest_fcn_x(self.N_test, self.X_quad)    # shape=(N_test, N_quad, 1)
        self.dtestfcn_y_quad_element = self.dTest_fcn_y(self.N_test, self.X_quad)    # shape=(N_test, N_quad, 1)

        # 计算变分形式的右端项（f与测试函数的积分）
        f_quad = self.f_ext(self.X_quad[:, 0], self.X_quad[:, 1])
        # 确保 f_quad 是列向量 (N_quad, 1)
        f_quad = f_quad.reshape([-1, 1])
        # 确保 w_quad 是列向量 (N_quad, 1)
        w_quad = self.w_quad.reshape([-1, 1])
        wf = w_quad * f_quad  # (N_quad, 1)
        self.var_f_ext = paddle.mm(self.testfcn_quad_element, wf).squeeze().reshape([-1])

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

    def Test_fcn(self, n, X):
        # 二维测试函数：phi_i(x) * phi_j(y)
        x = X[:, 0].numpy() if isinstance(X, paddle.Tensor) else X[:, 0]
        y = X[:, 1].numpy() if isinstance(X, paddle.Tensor) else X[:, 1]
        test_total = []
        for n in range(1, self.N_test + 1):
            test_x = Jacobi(n + 1, 0, 0, x) - Jacobi(n - 1, 0, 0, x)
            test_y = Jacobi(n + 1, 0, 0, y) - Jacobi(n - 1, 0, 0, y)
            test_total.append(test_x * test_y)
        test_total = np.array(test_total, dtype=np.float32)  # shape (N_test, N_quad)
        return paddle.to_tensor(test_total, dtype=d_type)

    def dTest_fcn_x(self, n, X):
        # 测试函数对x的偏导数：d(phi_i(x))/dx * phi_j(y)
        x = X[:, 0].numpy() if isinstance(X, paddle.Tensor) else X[:, 0]
        y = X[:, 1].numpy() if isinstance(X, paddle.Tensor) else X[:, 1]
        dtest_total = []
        for n in range(1, self.N_test + 1):
            # d/dx of (Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)) = DJacobi(n+1,0,0,x,1) - DJacobi(n-1,0,0,x,1)
            dtest_x = safe_DJacobi(n + 1, 0, 0, x, 1) - safe_DJacobi(n - 1, 0, 0, x, 1)
            test_y = Jacobi(n + 1, 0, 0, y) - Jacobi(n - 1, 0, 0, y)
            dtest_total.append(dtest_x * test_y)
        dtest_total = np.array(dtest_total, dtype=np.float32)  # shape (N_test, N_quad)
        return paddle.to_tensor(dtest_total, dtype=d_type)

    def dTest_fcn_y(self, n, X):
        # 测试函数对y的偏导数：phi_i(x) * d(phi_j(y))/dy
        x = X[:, 0].numpy() if isinstance(X, paddle.Tensor) else X[:, 0]
        y = X[:, 1].numpy() if isinstance(X, paddle.Tensor) else X[:, 1]
        dtest_total = []
        for n in range(1, self.N_test + 1):
            test_x = Jacobi(n + 1, 0, 0, x) - Jacobi(n - 1, 0, 0, x)
            dtest_y = safe_DJacobi(n + 1, 0, 0, y, 1) - safe_DJacobi(n - 1, 0, 0, y, 1)
            dtest_total.append(test_x * dtest_y)
        dtest_total = np.array(dtest_total, dtype=np.float32)  # shape (N_test, N_quad)
        return paddle.to_tensor(dtest_total, dtype=d_type)

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

        # 变分损失
        x_quad = self.X_quad[:, 0].reshape([-1, 1])
        y_quad = self.X_quad[:, 1].reshape([-1, 1])
        u_quad = self.forward(x_quad, y_quad)
        d1xu_quad, d1yu_quad, _, _ = self.net_du(x_quad, y_quad)

        # 确保形状
        u_quad = u_quad.reshape([-1, 1])
        d1xu_quad = d1xu_quad.reshape([-1, 1])
        d1yu_quad = d1yu_quad.reshape([-1, 1])
        w_quad = self.w_quad.reshape([-1, 1])

        # 变分形式中的扩散项： epsilon * ∫∇u·∇v dx = epsilon * ∫ (u_x * v_x + u_y * v_y) dx
        # 对流项： ∫ (b·∇u) v dx = ∫ (bx * u_x + by * u_y) * v dx
        # 因此 var_U_NN = epsilon * (∫ u_x * v_x + u_y * v_y) + ∫ (bx * u_x + by * u_y) * v

        # 计算 ∫ u_x * v_x
        w_u_x = w_quad * d1xu_quad  # (N_quad,1)
        var_diff_x = paddle.mm(self.dtestfcn_x_quad_element, w_u_x).squeeze()  # shape (N_test,)
        # 计算 ∫ u_y * v_y
        w_u_y = w_quad * d1yu_quad
        var_diff_y = paddle.mm(self.dtestfcn_y_quad_element, w_u_y).squeeze()
        var_diffusion = self.epsilon * (var_diff_x + var_diff_y)

        # 计算 ∫ (bx * u_x + by * u_y) * v
        w_convection = w_quad * (self.bx * d1xu_quad + self.by * d1yu_quad)
        var_convection = paddle.mm(self.testfcn_quad_element, w_convection).squeeze()

        self.var_U_NN = var_diffusion + var_convection  # shape (N_test,)

        # 确保 var_f_ext 和 var_U_NN 都是一维
        var_f_ext = self.var_f_ext.reshape([-1])
        var_U_NN = self.var_U_NN.reshape([-1])
        lossv = paddle.mean((var_f_ext - var_U_NN) ** 2)

        # 总损失
        lossb_weight = 100
        lossv_weight = 1
        loss = lossb_weight * lossb + lossad + lossv_weight * lossv
        return loss, lossb, lossad, lossv

    def train(self, N_iter, tresh):
        """
        训练函数
        """
        start_time = time.time()
        for it in range(N_iter):
            self.optimizer_Adam.clear_grad()
            loss, lossb, lossad, lossv = self.loss()
            if (it + 1) % 1000 == 0:
                time_now = time.time()
                time_cost = time_now - start_time
                start_time = time_now
                print("Iter: %d, Loss: %.3e, Lossb: %.3e, Lossad: %.3e, Lossv: %.3e" % (it+1, loss.numpy(), lossb.numpy(), lossad.numpy(), lossv.numpy()))
                print("Time cost: %.2f" % (time_cost))
            if loss < tresh:
                break

            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()


if __name__ == "__main__":
    """
    参数定义
    """
    LR = 0.001
    N_iter = 5000
    tresh = 2e-32
    Net_layer = [2] + [20] * 3 + [1]
    N_testfcn = 40
    N_quad = 100
    lossb_weight = 100
    lossv_weight = 1
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
        # f = -epsilon * Delta(u) + b · ∇u
        # 计算 Delta(u) = u_xx + u_yy = -2 * π^2 * sin(πx) * sin(πy)
        # 计算 b · ∇u = bx * u_x + by * u_y = 1 * (π cos(πx) sin(πy)) + 0 * (...)
        # 所以 f = -epsilon * (-2π^2 sin(πx) sin(πy)) + π cos(πx) sin(πy)
        #        = 2 * epsilon * π^2 * sin(πx) sin(πy) + π cos(πx) sin(πy)
        term1 = 2 * epsilon * (np.pi ** 2) * paddle.sin(np.pi * x) * paddle.sin(np.pi * y)
        term2 = np.pi * paddle.cos(np.pi * x) * paddle.sin(np.pi * y)
        return term1 + term2

    """
    数值积分方法和训练点的初始化
    """
    [X_quad_1d, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    Y_quad_1d, WY_quad = X_quad_1d, WX_quad
    xx, yy = np.meshgrid(X_quad_1d, Y_quad_1d)
    wxx, wyy = np.meshgrid(WX_quad, WY_quad)
    XY_quad = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    WXY_quad = (wxx * wyy).flatten()[:, None]

    # 边界点：方形区域四条边上的点
    n_bc = 100
    # 下边界
    bc_down_x = np.linspace(x_l, x_r, n_bc)
    bc_down_y = np.full(n_bc, y_l)
    # 上边界
    bc_up_x = np.linspace(x_l, x_r, n_bc)
    bc_up_y = np.full(n_bc, y_r)
    # 左边界
    bc_left_x = np.full(n_bc, x_l)
    bc_left_y = np.linspace(y_l, y_r, n_bc)
    # 右边界
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
    model = VPINN_CD(
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
    plt.title('Neural Network Prediction')
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
    plt.savefig('VPINN-ConvectionDiffusion_figure1.png', dpi=300)
    plt.show()