import math
import jax
import jax.numpy as jnp
from jax import random
import torch
import torch.nn.functional as F

# Sample from Gaussian processes
# Adapted from https://gist.github.com/neubig/e859ef0cc1a63d1c2ea4

def rbf_kernel(x1, x2, variance=1):
    return torch.exp(-1 * ((x1 - x2) ** 2) / (2*variance))

def ou_kernel(x1, x2, variance=1):
    return torch.exp(-1 * torch.abs(x1 - x2) / variance)

# xs: (N)
def gram_matrix(xs, kernel=rbf_kernel):
    x1 = xs.unsqueeze(1) # (N, 1)
    x2 = xs.unsqueeze(0) # (1, N)
    return kernel(x1, x2)

"""
# xs: (N)
# shape: (...)
# Output: (..., N)
def sample_gp(xs, kernel=rbf_kernel, shape=None, variance=1, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    kernel_var = lambda x1, x2: kernel(x1, x2, variance=variance)

    mean = torch.zeros(len(xs), device=xs.device, dtype=xs.dtype)
    gram = gram_matrix(xs, kernel_var)

    if shape is None:
        ys = torch.distributions.MultivariateNormal(mean, gram).sample()
    else:
        ys = torch.distributions.MultivariateNormal(mean, gram).sample(shape)
    
    return ys
"""

# xs: (N)
# shape: (...)
# Output: (..., N)
def sample_gp(xs, kernel=rbf_kernel, shape=None, variance=1, seed=None):
    
    # Convert torch tensor to JAX array using dlpack
    xs_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(xs))
    
    # Create JAX kernel
    def kernel_jax(x1, x2, variance=variance):
        return jnp.exp(-1 * ((x1 - x2) ** 2) / (2*variance))
    
    # Compute gram matrix using JAX
    x1 = xs_jax[:, None]
    x2 = xs_jax[None, :]
    gram = kernel_jax(x1, x2)
    
    # Setup JAX random state
    if seed is not None:
        key = random.PRNGKey(seed)
    else:
        key = random.PRNGKey(0)
    
    # Sample using JAX's multivariate_normal
    mean = jnp.zeros(len(xs_jax))
    if shape is None:
        ys_jax = random.multivariate_normal(
            key, mean, gram, 
            method='svd'  # Use SVD method for better stability
        )
    else:
        ys_jax = random.multivariate_normal(
            key, mean, gram,
            shape=shape,
            method='svd'
        )
    
    # Convert back to torch using dlpack
    ys = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(ys_jax))
    
    return ys

# eqn_class 0 = all terms
# eqn_class 1 = no a_2 term
# eqn_class 2 = no a_1 term
# size: (...)
# Output: a1, a2, a3, a4 = (...)
def sample_params(eqn_class=1, scale=1, size=None, device="cuda"):
    if size is None:
        size = []

    if eqn_class != 2:
        a1 = torch.empty(size, device=device).uniform_(1-0.5*scale, 1+0.5*scale)
    else:
        a1 = torch.zeros(size, device=device)
    
    if eqn_class != 1:
        a2 = torch.empty(size, device=device).uniform_(1-0.5*scale, 1+0.5*scale)
    else:
        a2 = torch.zeros(size, device=device)
    
    a3 = torch.empty(size, device=device).uniform_(-scale, scale)
    a4 = torch.empty(size, device=device).uniform_(-scale, scale)
    
    return a1, a2, a3, a4

# f: (..., N)
# x_eval: (N_eval)
# Output: f_eval: (..., N_eval)
def interpolate_cheb(f, x_eval, eps=1e-14):
    N = f.shape[-1]
    i = torch.linspace(0, 1, N, device=f.device, dtype=f.dtype)
    to_cheb = lambda x: torch.cos(math.pi*x)
    x = to_cheb(i).to(device=f.device) # Chebyshev points

    # Define weights
    w_x = torch.zeros(N, device=f.device, dtype=f.dtype)
    w_x[::2] = 1
    w_x[1::2] = -1
    w_x[0] = 0.5
    w_x[-1] = 0.5

    # Define difference matrix
    d_x = x_eval.unsqueeze(0) - x.unsqueeze(1) # (N, N_eval)

    # Handle numerical instability for small differences
    small_diff = torch.abs(d_x) < eps
    small_diff_max = torch.max(small_diff, dim=0).values
    # If small_diff, set the column to 0 and the entry to 1
    d_x = torch.where(
        small_diff_max[None, :],
        torch.zeros_like(d_x),
        1.0 / d_x
    )
    d_x[small_diff] = 1 # assumes no column has multiple small_diff

    # Interpolate
    f_eval_num = torch.einsum("...n,nm,n->...m", f, d_x, w_x)
    f_eval_denom = torch.einsum("nm,n->m", d_x, w_x)

    return f_eval_num / f_eval_denom

# n: int
# x: (n)
# Output: (n, n)
def cheb(n, x=None, device="cuda", dtype=torch.float32):
    if x is None:
        t = torch.linspace(0, 1, n, dtype=dtype, device=device)
        x = torch.cos(torch.pi * t)

    # Initialize D matrix
    D = torch.zeros((n, n), dtype=dtype, device=device)
    
    # Handle the corner elements first
    D[0, 0] = (2*(n-1)**2 + 1) / 6
    D[n-1, n-1] = -(2*(n-1)**2 + 1) / 6
    
    # Compute c arrays (multipliers for endpoints)
    c = torch.ones(n, dtype=dtype, device=device)
    c[0] = 2
    c[n-1] = 2
    
    # Compute off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (c[i]/c[j]) * ((-1)**(i+j)) / (x[i] - x[j])
    
    # Compute remaining diagonal elements
    for i in range(1, n-1):
        D[i, i] = -D[i, :].sum()
    
    return D, x

def pseudospectral_solve_batch(a_1, a_2, a_3, a_4, u_0, c_1, c_2, do_reshape=True, device="cuda", dtype=torch.float32):

    N = c_1.shape[-1]
    
    if do_reshape:
        num_examples_shape = c_1.shape[:-1]
        a_1 = a_1.reshape(-1)
        a_2 = a_2.reshape(-1)
        a_3 = a_3.reshape(-1)
        a_4 = a_4.reshape(-1)
        u_0 = u_0.reshape(-1)
        c_1 = c_1.reshape(-1, N)
        c_2 = c_2.reshape(-1, N)
    
    num_examples = c_1.shape[0]
    
    # Get Chebyshev differentiation matrix
    D, _ = cheb(N, device=device, dtype=dtype)  # Returns (N, N)
    
    # Create e_n2 vector with N points
    e_n2 = torch.zeros(N, dtype=dtype, device=device)
    e_n2[(N-1)//2] = 1
    
    # Create diagonal matrices for c_2 (vectorized version of np.diag)
    c_2_diag = torch.zeros(num_examples, N, N, dtype=dtype, device=device)
    diag_indices = torch.arange(N, device=device)
    c_2_diag[:, diag_indices, diag_indices] = c_2
    
    # Compute right-hand side b
    ones = torch.ones((num_examples, N), dtype=dtype, device=device)
    b = (a_1.unsqueeze(1) * c_1 + 
         a_4.unsqueeze(1) * ones + 
         a_3.unsqueeze(1) * u_0.unsqueeze(1) * e_n2.unsqueeze(0)).to(dtype=dtype)
    
    # Compute system matrix M
    # Follow numpy version exactly:
    # M = D[None, ...] - a_3[:, None, None]*eye + a_3[:, None, None]*diag(e_n2) - a_2[:, None, None]*diag(c_2)
    eye = torch.eye(N, dtype=dtype, device=device)
    M = (D.unsqueeze(0) + 
         (-a_3.unsqueeze(1).unsqueeze(2) * eye.unsqueeze(0)) + 
         (a_3.unsqueeze(1).unsqueeze(2) * torch.diag(e_n2).unsqueeze(0)) + 
         (-a_2.unsqueeze(1).unsqueeze(2) * c_2_diag)).to(dtype=dtype)
    
    # Solve the system
    u = torch.linalg.solve(M, b)
    
    # Reshape output if needed
    if do_reshape:
        u = u.reshape(*num_examples_shape, -1)
    
    return u