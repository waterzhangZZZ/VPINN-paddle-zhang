import paddle
import paddle.nn as nn
import numpy as np
from scipy.special import eval_legendre
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights, Jacobi
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

class VPINN(paddle.nn.Layer):
    def __init__(
        self,
        x_bc,
        u_bc,
        x_quad,
        w_quad,
        N_test,
        layers
    ):
        super(VPINN, self).__init__()

        # 定义网络结构
        paddle.set_default_dtype(d_type)
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(Sin())
        self.net = nn.Sequential(*net_layers)
        self.LR = LR
        self.optimizer_Adam = paddle.optimizer.Adam(
            parameters=self.parameters(), learning_rate=self.LR)

        self.x_bc = paddle.to_tensor(x_bc, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.u_bc = paddle.to_tensor(u_bc, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.x_quad = paddle.to_tensor(x_quad, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.w_quad = paddle.to_tensor(w_quad, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.N_quad = self.x_quad.shape[0]
        self.N_test = N_test
        self.x_train = paddle.linspace(-1, 1,
                                          self.x_quad.shape[0],
                                          dtype=d_type).reshape([-1, 1])
        self.x_train.stop_gradient = False
        self.testfcn_quad_element = self.Test_fcn(self.N_test, self.x_quad).astype(d_type) # shape=(N_test, N_quad, 1)
        self.d1testfcn_quad_element = self.dTest_fcn(self.N_test, self.x_quad)[0].astype(d_type) # shape=(N_test, N_quad, 1)
        self.d2testfcn_quad_element = self.dTest_fcn(self.N_test, self.x_quad)[1].astype(d_type) # shape=(N_test, N_quad, 1)        
        self.var_f_ext = paddle.stack([
            paddle.sum(self.w_quad * f_ext(self.x_quad) * self.testfcn_quad_element[i][:].reshape(-1,1))
            for i in range(self.N_test)
        ])

    def forward(self, x):
        """前向传播"""
        if isinstance(x, np.ndarray):
            with paddle.no_grad():
                x = paddle.to_tensor(x, dtype=d_type)
                u_pred = self.net_u(x)
                return u_pred.numpy()
        else:    
            x.stop_gradient = False  # 确保x是可求导的
            u = self.net(x)
            return u

    def net_u(self, x):
        return self.net(x)
    
    def net_du(self, x):
        """计算u对x的一阶和二阶导数"""
        x.stop_gradient = False  
        u = self.forward(x)
        
        # 计算一阶导数
        d1u = paddle.grad(u, x, grad_outputs=paddle.ones_like(u),
                               create_graph=True)[0]
        
        # 计算二阶导数
        d2u = paddle.grad(d1u, x, grad_outputs=paddle.ones_like(d1u),
                                create_graph=True)[0]
        
        return d1u, d2u
    
    def Test_fcn(self, n, x):
        # 返回x点处，所有测试函数的值，n暂时没有作用
        test_total = []
        for n in range(1, self.N_test + 1):
            test = Jacobi(n + 1, 0, 0, x) - Jacobi(n - 1, 0, 0, x)
            test_total.append(test)
        return np.asarray(test_total)
    def dTest_fcn(self, N_test, x):
            d1test_total = []
            d2test_total = []
            for n in range(1, N_test + 1):
                if n == 1:
                    d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x)
                    d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x)
                    d1test_total.append(d1test)
                    d2test_total.append(d2test)
                elif n == 2:
                    d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x) - n / 2 * Jacobi(
                        n - 2, 1, 1, x
                    )
                    d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x)
                    d1test_total.append(d1test)
                    d2test_total.append(d2test)
                else:
                    d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x) - n / 2 * Jacobi(
                        n - 2, 1, 1, x
                    )
                    d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x) - n * (
                        n + 1
                    ) / (2 * 2) * Jacobi(n - 3, 2, 2, x)
                    d1test_total.append(d1test)
                    d2test_total.append(d2test)
            return np.asarray(d1test_total), np.asarray(d2test_total)

    def loss(self):
        u_bc_pred = self.net_u(self.x_bc)
        lossb = paddle.mean((u_bc_pred - self.u_bc) ** 2)

        d1u_train_pred, d2u_train_pred = self.net_du(self.x_train)
        f_train = f_ext(self.x_train)
        lossad = paddle.mean((d2u_train_pred + f_train) ** 2)

        # 实现lossv，先尝试对测试函数进行循环，对于每个积分点计算数值积分，var_U_NN,var_f_ext
        # var_F_ext可以在__init__中实现
        self.var_U_NN = paddle.stack([
            paddle.sum(self.w_quad * self.forward(self.x_quad) * self.testfcn_quad_element[i][:].reshape(-1,1))
            for i in range(self.N_test)
        ])
        # self.var_U_NN = paddle.stack([
        #     paddle.sum(self.w_quad * (-self.net_du(self.x_quad)[1]) * self.testfcn_quad_element[i][:].reshape(-1,1))
        #     for i in range(self.N_test)
        # ])

        # self.var_U_NN = paddle.stack([
        #     paddle.sum(self.w_quad * (-self.net_du(self.x_quad)[0]) * self.d1testfcn_quad_element[i][:].reshape(-1,1))
        #     for i in range(self.N_test)
        # ])
        lossv = paddle.mean((self.var_f_ext - self.var_U_NN)**2)
        
        loss = lossb_weight * lossb + lossv_weight * lossv + lossad
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
            # 更新参数
            
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
        

if __name__ == "__main__":
    """
    参数定义
    """
    LR = 0.001
    N_iter = 500
    tresh = 2e-32
    var_form = 1
    Net_layer = [1] + [20] * 3 + [1]
    N_testfcn = 40
    N_quad = 100
    lossb_weight = 100
    lossv_weight = 1
    x_l, x_r = -1.0, 1.0

    """
    u_ext:方程的解析解
    f_ext:方程的外部力定义
    """
    # def u_ext(x):
    #     uext = x**3 / 6 - x**4 / 12
    #     return uext
    
    # def f_ext(x):
    #     fext = -x*(1-x)
    #     return fext
    
    omega = 8 * np.pi
    amp = 1  #解的振幅
    r1 = 80  #和解的形状有关的参数

    def u_ext(x):
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        utemp = 0.1 * np.sin(omega * x) + np.tanh(r1 * x)
        return amp * paddle.to_tensor(utemp, dtype=d_type)
    def f_ext(x):
        """泊松方程的源项：f(x) = -d²u/dx²"""
        if isinstance(x, paddle.Tensor):
            x = x.detach().numpy()
        gtemp = -(-0.1 * omega**2 * np.sin(omega * x) - 2 * r1**2 * (1/np.cosh(r1 * x))**2 * np.tanh(r1 * x))
        return amp * paddle.to_tensor(gtemp, dtype=d_type)
        
    """
    数值积分方法和训练点的初始化
    """
    [x_quad, w_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    # x_quad: (-1,1)区间的积分点
    # x_quad_ext: (x_l,x_r)区间的积分点
    x_quad_ext = x_l + (x_r - x_l) / 2 * (x_quad + 1)
    x_bc = np.array([x_l, x_r])
    u_bc = [u_ext(x_l), u_ext(x_r)]

    loss_history = []
    model = VPINN(
        x_bc = x_bc,
        u_bc = u_bc,
        x_quad = x_quad_ext,
        w_quad = w_quad,
        N_test = N_testfcn,
        layers = Net_layer,
    )
    model.train(N_iter, tresh)
    # 计算范围内的平均残差
    residual_point = paddle.linspace(x_l, x_r, 100, dtype=d_type).reshape([-1,1])
    u_nn_point = model.forward(residual_point)
    u_ext_point = u_ext(residual_point)

    residual = paddle.mean((u_nn_point - u_ext_point)**2)
    print("average residual:", residual)


    #++++++++++++++++++++++++++++
    # Test point
    delta_test = 0.001  # 表示测试点之间的间隔。
    xtest = np.arange(-1, 1 + delta_test, delta_test)   # 这个数列定义了测试点在空间上的位置。
    data_temp = np.asarray([[xtest[i], u_ext(xtest[i])] for i in range(len(xtest))])
    X_test = data_temp.flatten()[0::2]  # 将 data_temp 数组展平，偶数索引元素，这些是测试点的坐标
    u_test = data_temp.flatten()[1::2]  # 奇数索引元素，这些是在测试点上的解的值
    X_test = X_test[:, None]    # 通过增加一个维度，将 X_test 转换为列向量，以匹配神经网络输入数据的期望格式。
    u_test = u_test[:, None]
    f_test = f_ext(X_test)  # 计算 X_test 中每个点上的外部力或源项
    u_pred = model.forward(X_test)  # 使用训练好的模型预测 X_test 中每个点的解
    filename = ("D:\code\HVPINN_zhang\hp-VPINNs-paddle\\complex_Sin_b+ad+v_test.pdf")

    font = 24
    pnt_skip = 25
    plt.grid(True)
    fig, ax = plt.subplots()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=8)
    plt.xlabel("$x$", fontsize=font)
    plt.ylabel("$u$", fontsize=font)
    plt.axhline(0, linewidth=0.8, linestyle="-", color="gray")
    for xc in [x_l, x_r]:
        plt.axvline(x=xc, linewidth=2, ls="--")
    plt.plot(X_test, u_test, linewidth=1, color="r", label="".join(["$exact$"]))
    plt.plot(X_test[0::pnt_skip], u_pred[0::pnt_skip], "k*", label="$hp-VPINN$")
    plt.tick_params(labelsize=20)
    legend = plt.legend(shadow=True, loc="upper left", fontsize=18, ncol=1)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(filename)


        # 绘制结果
    with paddle.no_grad():
        x_test = paddle.linspace(x_l, x_r, 200, dtype=d_type).reshape([-1, 1])
        u_pred = model.forward(x_test)
        u_true = u_ext(x_test)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_test.numpy(), u_true.numpy(), 'r-', label='Exact Solution')
        plt.plot(x_test.numpy(), u_pred.numpy(), 'b--', label='PINN Prediction')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
        plt.title('PINN Solution for Poisson Equation')
        plt.show()
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()