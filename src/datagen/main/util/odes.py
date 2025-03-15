from math import exp
import numpy as np


# Sample from Gaussian processes
# Adapted from https://gist.github.com/neubig/e859ef0cc1a63d1c2ea4


def rbf_kernel(x1, x2, variance=1):
    return exp(-1 * ((x1 - x2) ** 2) / (2 * variance))


def ou_kernel(x1, x2, variance=1):
    return exp(-1 * abs(x1 - x2) / variance)


def gram_matrix(xs, kernel=rbf_kernel):
    return [[kernel(x1, x2) for x2 in xs] for x1 in xs]


# Gaussian process defined on xs
def sample_gp(xs, kernel=rbf_kernel, shape=None, variance=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    kernel_var = lambda x1, x2: kernel(x1, x2, variance=variance)

    mean = [0 for _ in xs]
    gram = gram_matrix(xs, kernel_var)

    if shape is None:
        ys = rng.multivariate_normal(mean, gram)
    else:
        ys = rng.multivariate_normal(mean, gram, shape)

    return ys


def sample_params(eqn_class=1, scale=1, size=None, rng=None):
    """
    Sample the parameters of the ODE:
    u'(x) = a₁c₁(x) + a₂c₂(x)u(x) + a₃u(x) + a₄
    with initial condition u(0) = u₀.

    a1, a2, a3, a4 = (num_examples)
    a1 sampled from Uniform(1 - 0.5 * scale, 1 + 0.5 * scale)
    a2 sampled from Uniform(1 - 0.5 * scale, 1 + 0.5 * scale)
    a3 sampled from Uniform(-scale, scale)
    a4 sampled from Uniform(-scale, scale)

    eqn_class 0 = all terms
    eqn_class 1 = no a2 term
    eqn_class 2 = no a1 term
    """
    if rng is None:
        rng = np.random.default_rng()

    if eqn_class != 2:
        a1 = rng.uniform(1 - 0.5 * scale, 1 + 0.5 * scale, size=size)
    else:
        a1 = np.zeros(size)
    if eqn_class != 1:
        a2 = rng.uniform(1 - 0.5 * scale, 1 + 0.5 * scale, size=size)
    else:
        a2 = np.zeros(size)
    a3 = rng.uniform(-scale, scale, size=size)
    a4 = rng.uniform(-scale, scale, size=size)

    return a1, a2, a3, a4


def interpolate_cheb(f, x_eval):
    """
    Interpolate given function evaluated on Chebyshev points (of the 2nd kind)
    f = (... N+1)
    x_eval = (N_eval)
    Output: f_eval = (... N_eval)
    """
    N = f.shape[-1] - 1
    i = np.linspace(0, 1, N + 1)
    to_cheb = lambda x: np.cos(np.pi * x)
    x = to_cheb(i)  # Chebyshev points

    # Define weights
    w_x = np.zeros(N + 1)  # (N+1)
    w_x[::2] = 1
    w_x[1::2] = -1
    w_x[0] /= 2
    w_x[-1] /= 2

    # Define differences
    d_x = x_eval[None, ...] - x[..., None]  # (N+1, n_eval)
    for eval_i in range(x_eval.shape[0]):
        arg_j = np.argmin(np.abs(d_x[..., eval_i]))
        if np.abs(d_x[arg_j, eval_i]) < 1e-14:
            d_x[..., eval_i] = 0
            d_x[arg_j, eval_i] = 1
        else:
            d_x[..., eval_i] = 1 / d_x[..., eval_i]

    f_eval_num = np.einsum("...n,nm,n->...m", f, d_x, w_x)
    f_eval_denom = np.einsum("nm,n->m", d_x, w_x)
    return np.einsum("...m,m->...m", f_eval_num, 1 / f_eval_denom)


def cheb(n, x=None):
    """
    Manually define the spectral differentiation matrix on n+1 Chebyshev nodes
    n = number of Chebyshev nodes
    x = (n+1) Chebyshev nodes
    D = (n+1, n+1) spectral differentiation matrix
    """
    if x is None:
        to_cheb = lambda x: np.cos(np.pi * x)
        x = to_cheb(np.linspace(0, 1, n + 1))

    D = np.zeros((n + 1, n + 1))
    # Compute entries of D except for diag[1] through diag[n-1]
    for i in range(n + 1):
        for j in range(n + 1):
            c_i = 2 if (i == 0 or i == n) else 1
            c_j = 2 if (j == 0 or j == n) else 1
            if i == 0 and j == 0:
                D[i, j] = (2 * n**2 + 1) / 6
            elif i == n and j == n:
                D[i, j] = -(2 * n**2 + 1) / 6
            elif i != j:
                D[i, j] = c_i / c_j * (-1) ** (i + j) / (x[i] - x[j])

    # For numerical accuracy, compute the remaining diagonal entries
    for i in range(1, n):
        D[i, i] = -np.sum(D[i, :])

    return D, x


def pseudospectral_solve_batch(a_1, a_2, a_3, a_4, u_0, c_1, c_2, do_reshape=True):
    """
    Solve the ODE:
        u'(x) = a₁c₁(x) + a₂c₂(x)u(x) + a₃u(x) + a₄
        with initial condition u(0) = u₀
    using the pseudospectral method.

    c_1 = (num_examples, N+1)
    c_2 = (num_examples, N+1)
    u_0 = (num_examples)
    u = (num_examples, N+1)
    a1, a2, a3, a4 = (num_examples)
    """
    # Assumes N is even when enforcing the initial condition
    N = c_1.shape[-1] - 1
    if do_reshape:
        num_examples_shape = c_1.shape[:-1]
        a_1 = a_1.reshape(-1)
        a_2 = a_2.reshape(-1)
        a_3 = a_3.reshape(-1)
        a_4 = a_4.reshape(-1)
        u_0 = u_0.reshape(-1)
        c_1 = c_1.reshape(-1, N + 1)
        c_2 = c_2.reshape(-1, N + 1)
    num_examples = c_1.shape[0]
    D, _ = cheb(N)  # (N+1, N+1)

    vectdiag = np.vectorize(np.diag, signature="(n)->(n,n)")

    # Solve linear system
    e_n2 = np.zeros(N + 1)
    e_n2[N // 2] = 1  # (N+1)
    b = (
        a_1[:, None] * c_1
        + a_4[:, None] * np.ones((num_examples, N + 1))
        + a_3[:, None] * u_0[:, None] * e_n2[None, :]
    )  # (num_examples, N+1)
    M = (
        D[None, ...]
        - a_3[:, None, None] * np.eye(N + 1)[None, ...]
        + a_3[:, None, None] * np.diag(e_n2)[None, ...]
        - a_2[:, None, None] * vectdiag(c_2)
    )  # (num_examples, N+1, N+1)
    u = np.linalg.solve(M, np.expand_dims(b, axis=-1)).squeeze(-1)

    if do_reshape:
        u = u.reshape(*num_examples_shape, -1)
    return u
