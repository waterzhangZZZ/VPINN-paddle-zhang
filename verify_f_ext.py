import paddle
import numpy as np

def u_ext(x, y):
    u = (0.1 * paddle.sin(np.pi * x) + paddle.tanh(10 * x)) * paddle.sin(np.pi * y)
    return u

def f_ext_original(x, y):
    # Original implementation from VPINN_Possion.py
    f = (-0.1 * np.pi**2 * paddle.sin(np.pi * x) - 200 * paddle.tanh(10 * x) * paddle.sin(np.pi * y)) \
        / paddle.cosh(100 * x)**2 + (0.1 * paddle.sin(np.pi * x) + paddle.tanh(10 * x)) * (-np.pi**2 * paddle.sin(np.pi * y))
    return f

def get_exact_f(x, y):
    x.stop_gradient = False
    y.stop_gradient = False
    u = u_ext(x, y)
    
    d1xu = paddle.grad(u, x, grad_outputs=paddle.ones_like(u), create_graph=True)[0]
    d1yu = paddle.grad(u, y, grad_outputs=paddle.ones_like(u), create_graph=True)[0]
    
    d2xu = paddle.grad(d1xu, x, grad_outputs=paddle.ones_like(d1xu), create_graph=True)[0]
    d2yu = paddle.grad(d1yu, y, grad_outputs=paddle.ones_like(d1yu), create_graph=True)[0]
    
    # Poisson equation: -Laplacian(u) = f
    # f = -(u_xx + u_yy)
    f = -(d2xu + d2yu)
    return f

# Test points
x = paddle.to_tensor(np.linspace(-1, 1, 10).astype('float32').reshape(-1, 1))
y = paddle.to_tensor(np.linspace(-1, 1, 10).astype('float32').reshape(-1, 1))
# Meshgrid
x_grid, y_grid = paddle.meshgrid(x.flatten(), y.flatten())
x_flat = x_grid.flatten().reshape([-1, 1])
y_flat = y_grid.flatten().reshape([-1, 1])

f_exact = get_exact_f(x_flat, y_flat)
f_orig = f_ext_original(x_flat, y_flat)

diff = paddle.abs(f_exact - f_orig)
print(f"Max difference between exact f and original f_ext: {paddle.max(diff).item()}")
print(f"Mean difference: {paddle.mean(diff).item()}")

# Print some values
print("\nSample values:")
print(f"Exact f at (0,0): {get_exact_f(paddle.to_tensor([0.0]), paddle.to_tensor([0.0])).item()}")
print(f"Orig f at (0,0): {f_ext_original(paddle.to_tensor([0.0]), paddle.to_tensor([0.0])).item()}")

# Check if f_orig is actually Delta u (sign flip)
diff_sign = paddle.abs(f_exact + f_orig)
print(f"\nMax difference if sign flipped: {paddle.max(diff_sign).item()}")
