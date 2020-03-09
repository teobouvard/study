import numpy as np


def to_latex(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += [
        "  " + " & ".join([f"{float(n):.3f}" for n in l.split()]) + r" \\"
        for l in lines
    ]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


def activation(x):
    return 1 / (1 + np.exp(x))


def activation_prime(x):
    return activation(x) * (activation(x) - 1)


x_1 = np.array([1, 1, 1 / 4, 1 / 4]).T.reshape(-1, 1)
y_1 = np.array([1, 0, 0, 0]).T.reshape(-1, 1)
mu = 1

theta_11 = np.array([0.0, -0.5, 0.5])
theta_12 = np.array([0.5, -0.5, 0.0])
theta_13 = np.array([-0.5, 0.0, 0.5])
bias_1 = np.array([0.5, -0.5, 0.5])

theta_1 = np.vstack([theta_11, theta_12, theta_13])
theta_1 = np.column_stack([bias_1, theta_1])

theta_21 = np.array([0.5, -0.5, 0.5])
theta_22 = np.array([0.0, -0.5, 0.5])
theta_23 = np.array([-0.5, 0.5, 0.0])
theta_24 = np.array([0.5, 0.0, -0.5])
bias_2 = np.array([-0.5, 0.5, -0.5, 0.5])

theta_2 = np.vstack([theta_21, theta_22, theta_23, theta_24])
theta_2 = np.column_stack([bias_2, theta_2])


y_11 = theta_1 @ x_1

print("Weight matrix\n", theta_1)
print("Training vector\n", x_1)
print("Multiplication output\n", y_11)

y_11 = activation(y_11)

print("Activation pass\n", y_11)

y_11 = np.vstack([[1], y_11])
y_21 = theta_2 @ y_11

print("Weight matrix\n", theta_2)
print("Hidden vector\n", y_11)
print("Multiplication output\n", y_21)

y_21 = activation(y_21)

print("Output layer\n", y_21)

loss = 0.5 * np.linalg.norm(y_21 - y_1) ** 2

print("J(theta)\n", loss)

sigprime = activation_prime(theta_2 @ y_11)
print(f"{sigprime=}")

delta_2 = np.multiply((y_21 - y_1), sigprime)
print(f"{delta_2=}")

delta_theta_2 = -mu * delta_2 @ y_11.T

print(f"{delta_theta_2=}")

new_theta_2 = theta_2 + delta_theta_2
print(f"{theta_2=}")
print(to_latex(new_theta_2))
