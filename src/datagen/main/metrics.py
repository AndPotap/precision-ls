def squared_error(y_pred, y):
    return (y - y_pred).square()


def mean_squared_error(y_pred, y):
    return (y - y_pred).square().mean()


def mean_absolute_error(y_pred, y):
    return (y - y_pred).abs().mean()


def relative_error(y_pred, y):
    return (y - y_pred).abs() / y.abs()


def mean_relative_error(y_pred, y):
    return (y - y_pred).abs().mean() / y.abs().mean()


def mean_relative_l2_error(y_pred, y):
    return (y - y_pred).square().mean().sqrt() / y.abs().mean()
