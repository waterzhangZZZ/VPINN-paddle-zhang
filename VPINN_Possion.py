# 尝试一下批量处理方法？现在的方法是一次性计算所有点的平均误差
# 或许可以一次训练十个点，计算十个点的误差然后反向传播
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
        X_bc,           # 二维问题边界取点集合，shape=[num_bc, 2]
        u_bc,           # x_bc取点处真实值（根据边界条件）提前计算，shape=[num_bc, 2]
        X_quad,
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
        
        self.x_bc = paddle.to_tensor(X_bc[:,0], dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.y_bc = paddle.to_tensor(X_bc[:,1], dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.u_bc = paddle.to_tensor(u_bc, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.x_quad = paddle.to_tensor(X_quad[:,0], dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.y_quad = paddle.to_tensor(X_quad[:,1], dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.w_quad = paddle.to_tensor(w_quad, dtype=d_type, stop_gradient=False).reshape([-1, 1])
        self.N_quad = self.x_quad.shape[0]
        self.N_test = N_test

        # 生成内部训练点（二维）
        n_train_point = 200         # 设置点的数量
        self.X_train = paddle.uniform(shape=[n_train_point, 2],
                                    min=-1, max=1, dtype=d_type)
        self.x_train = paddle.to_tensor(self.X_train[:,0], stop_gradient=False).reshape([-1, 1])
        self.y_train = paddle.to_tensor(self.X_train[:,1], stop_gradient=False).reshape([-1, 1])


        self.testfcn_quad_element = paddle.to_tensor(self.Test_fcn(self.N_test, self.x_quad, self.y_quad).astype(d_type)) # shape=(N_test, N_quad, 1)
        # self.d1testfcn_quad_element = self.dTest_fcn(self.N_test, self.x_quad)[0].astype(d_type) # shape=(N_test, N_quad, 1)
        # self.d2testfcn_quad_element = self.dTest_fcn(self.N_test, self.x_quad)[1].astype(d_type) # shape=(N_test, N_quad, 1)  
        # self.var_f_ext = paddle.stack([
        #     paddle.sum(self.w_quad * f_ext(self.x_quad, self.y_quad)
        #             * self.testfcn_quad_element[i][:].reshape(-1,1))
        #     for i in range(self.N_test)
        # ])
        f_quad = f_ext(self.x_quad, self.y_quad)
        self.var_f_ext = paddle.matmul(
            self.testfcn_quad_element.reshape([self.N_test, -1]),
            self.w_quad * f_quad
        ).squeeze()

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
        return self.net(paddle.concat([x, y], axis=1))
    
    ########################################################
    def net_dxu(self, x, y):
        x.stop_gradient = False
        u = self.forward(x, y)

        d1xu = paddle.grad(u, x, grad_outputs=paddle.ones_like(u),
                           create_graph=True)[0]
        d2xu = paddle.grad(d1xu, x, grad_outputs=paddle.ones_like(d1xu),
                           create_graph=True)[0]
        
        return d1xu, d2xu

    def net_dyu(self, x, y):
        x.stop_gradient = False
        u = self.forward(x, y)

        d1yu = paddle.grad(u, y, grad_outputs=paddle.ones_like(u),
                           create_graph=True)[0]
        d2yu = paddle.grad(d1yu, y, grad_outputs=paddle.ones_like(d1yu),
                           create_graph=True)[0]
        
        return d1yu, d2yu

    def Test_fcn(self, n, x, y):
    # 测试函数在二维平面中某点的值=这点x的测试函数值*这点y的测试函数值
        test_total_x = []
        for n in range(1, self.N_test + 1):
            test = Jacobi(n + 1, 0, 0, x) - Jacobi(n - 1, 0, 0, x)
            test_total_x.append(test)
        test_total_x = np.array(test_total_x)
        test_total_y = []
        for n in range(1, self.N_test + 1):
            test = Jacobi(n + 1, 0, 0, y) - Jacobi(n - 1, 0, 0, y)
            test_total_y.append(test)
        test_total_y = np.array(test_total_y)

        test_total=test_total_x*test_total_y    
        return np.asarray(test_total)

    # def dTest_fcn(self, N_test, x):
    #         d1test_total = []
    #         d2test_total = []
    #         for n in range(1, N_test + 1):
    #             if n == 1:
    #                 d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x)
    #                 d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x)
    #                 d1test_total.append(d1test)
    #                 d2test_total.append(d2test)
    #             elif n == 2:
    #                 d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x) - n / 2 * Jacobi(
    #                     n - 2, 1, 1, x
    #                 )
    #                 d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x)
    #                 d1test_total.append(d1test)
    #                 d2test_total.append(d2test)
    #             else:
    #                 d1test = (n + 2) / 2 * Jacobi(n, 1, 1, x) - n / 2 * Jacobi(
    #                     n - 2, 1, 1, x
    #                 )
    #                 d2test = (n + 2) * (n + 3) / (2 * 2) * Jacobi(n - 1, 2, 2, x) - n * (
    #                     n + 1
    #                 ) / (2 * 2) * Jacobi(n - 3, 2, 2, x)
    #                 d1test_total.append(d1test)
    #                 d2test_total.append(d2test)
    #         return np.asarray(d1test_total), np.asarray(d2test_total)

    def loss(self):
        u_bc_pred = self.net_u(self.x_bc, self.y_bc)
        lossb = paddle.mean((u_bc_pred - self.u_bc)**2)
        
        d1xu_train_pred, d2xu_train_pred = self.net_dxu(self.x_train, self.y_train)
        d1yu_train_pred, d2yu_train_pred = self.net_dyu(self.x_train, self.y_train)
        f_train = f_ext(self.x_train, self.y_train)
        lossad = paddle.mean((d2xu_train_pred+d2yu_train_pred+f_train)**2)

        # self.var_U_NN = paddle.stack([
        #     paddle.sum(self.w_quad * self.forward(self.x_quad, self.y_quad)
        #                * self.testfcn_quad_element[i][:].reshape(-1,1))
        #     for i in range(self.N_test)
        # ])
        u_quad = self.forward(self.x_quad, self.y_quad)
        self.var_U_NN = paddle.matmul(
            self.testfcn_quad_element.reshape([self.N_test, -1]),
            self.w_quad * u_quad
        ).squeeze()

        lossv = paddle.mean((self.var_f_ext - self.var_U_NN)**2)
        loss = lossb_weight * lossb + lossad# + lossv_weight * lossv
        return loss, lossb, lossad, lossv

    def train(self, N_iter, tresh):
        """
        训练函数
        """
        print("开始训练......")
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
    N_iter = 2000
    tresh = 2e-32
    var_form = 1
    Net_layer = [2] + [5] * 3 + [1]
    N_testfcn = 40
    N_quad = 100
    lossb_weight = 100
    lossv_weight = 1
    x_l, x_r = -1.0, 1.0
    y_l, y_r = -1.0, 1.0


    #################################################################################
    def u_ext(x, y):
        u = (0.1*paddle.sin(np.pi*x)+paddle.tanh(10*x)) * paddle.sin(np.pi*y)
        return u


    def f_ext(x, y):
        f = (-0.1*np.pi**2*paddle.sin(np.pi*x)-200*paddle.tanh(10*x)*paddle.sin(np.pi*y))\
            /paddle.cosh(100*x)**2 + (0.1*paddle.sin(np.pi*x)+paddle.tanh(10*x))*(-np.pi**2*paddle.sin(np.pi*y))
        return f

    """
    数值积分方法和训练点的初始化
    """
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    Y_quad, WY_quad   = (X_quad, WX_quad)
    xx, yy            = np.meshgrid(X_quad,  Y_quad)
    wxx, wyy          = np.meshgrid(WX_quad, WY_quad)
    XY_quad     = np.hstack((xx.flatten()[:,None],  yy.flatten()[:,None]))
    WXY_quad    = (wxx * wyy).flatten()[:, None]
    # 这个方程没有找到边界条件
    # 根据结果反推边界条件：y=1时，u=0
    bc_up_x = bc_down_x = np.linspace(-1, 1, 100)
    bc_up_y = np.ones(100)
    bc_down_y = np.full(100, -1)
    bc_left_x = np.full(100, -1)
    bc_right_x = np.ones(100)
    bc_left_y = bc_right_y = np.linspace(-1, 1, 100)
    
    X_bc = np.zeros([400,2])
    X_bc[0:100, 0]=bc_up_x
    X_bc[100:200, 0]=bc_down_x
    X_bc[200:300, 0]=bc_left_x
    X_bc[300:400, 0]=bc_right_x
    X_bc[0:100, 1]=bc_up_y
    X_bc[100:200, 1]=bc_down_y
    X_bc[200:300, 1]=bc_left_y
    X_bc[300:400, 1]=bc_right_y

    u_bc_up = u_bc_down = np.zeros(100)
    u_bc_left = -1*np.sin(np.pi*np.linspace(-1, 1, 100))
    u_bc_right = np.sin(np.pi*np.linspace(-1, 1, 100))
    u_bc = np.concat([u_bc_up, u_bc_down, u_bc_left, u_bc_right], axis=0)


    loss_history = []
    model = VPINN(
        X_bc = X_bc,
        u_bc = u_bc,
        X_quad = XY_quad,
        w_quad = WXY_quad,
        N_test = N_testfcn,
        layers = Net_layer,
    )   
    model.train(N_iter, tresh)

    ##################################################################################
    # 计算训练过程中loss变化，以及绘图计算参差
# 在训练完成后添加以下代码
def plot_results(model):
    # 生成测试网格点
    n_plot = 100
    x_plot = np.linspace(-1, 1, n_plot)
    y_plot = np.linspace(-1, 1, n_plot)
    xx, yy = np.meshgrid(x_plot, y_plot)
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # 计算预测值和真实值
    with paddle.no_grad():
        # 修正输入格式问题
        u_pred = model(
            paddle.to_tensor(grid_points[:,0], dtype=d_type).reshape([-1,1]),
            paddle.to_tensor(grid_points[:,1], dtype=d_type).reshape([-1,1])
        )
        u_pred = u_pred.numpy().reshape(n_plot, n_plot)
    
    # 计算真实值（使用NumPy避免Tensor问题）
    u_true_np = (0.1*np.sin(np.pi*xx) + np.tanh(10*xx)) * np.sin(np.pi*yy)
    
    # 计算误差（全部使用NumPy数组）
    error = np.abs(u_pred - u_true_np)
    
    # 创建绘图
    plt.figure(figsize=(18, 5))
    
    # 1. 神经网络预测值
    plt.subplot(1, 3, 1)
    plt.pcolormesh(xx, yy, u_pred, shading='auto', cmap='viridis')
    plt.colorbar(label='Predicted Value')
    plt.title('Neural Network Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 2. 真实解
    plt.subplot(1, 3, 2)
    plt.pcolormesh(xx, yy, u_true_np, shading='auto', cmap='viridis')
    plt.colorbar(label='True Value')
    plt.title('Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 3. 误差分布热力图
    plt.subplot(1, 3, 3)
    error_plot = plt.pcolormesh(xx, yy, error, shading='auto', cmap='hot_r')
    plt.colorbar(error_plot, label='Absolute Error')
    plt.title('Error Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig('vpinn_results.png', dpi=300)
    plt.show()
    
    # 打印最大误差和平均误差
    max_error = np.max(error)
    avg_error = np.mean(error)
    print(f"Maximum absolute error: {max_error:.4e}")
    print(f"Average absolute error: {avg_error:.4e}")

# 在训练完成后调用绘图函数
plot_results(model)