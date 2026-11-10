import numpy as np
# --- Ejercicio S5.1: forward con capa oculta ---
def sigmoide(z):
    return 1/(1 + np.exp(-z))
def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    h = sigmoide(z1)
    z2 = h @ W2 + b2
    a = sigmoide(z2)
    return h, a

# --- Ejercicio S5.2: comprobar el forward con pesos a mano ---
# --- Ejercicio S5.2: comprobar el forward con pesos a mano ---
W1 = np.array([[20, 20], [20, 20]])
b1 = np.array([-10, -30])
W2 = np.array([20, -20])
b2 = -10
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_xor = np.array([0,1,1,0])

for x, y in zip(X, Y_xor):
    h, a = forward(x, W1, b1, W2, b2)
    print(x, "->", round(float(a), 3), "esperado:", y)

# --- Ejercicio S5.3: gradientes de la capa de salida ---
def gradientes(entrada, a, y):
    dz = 2*(a - y) * a * (1-a)
    dw = dz * entrada
    db = dz
    return dw, db

dW2, db2 = gradientes(h, a, y)

# --- Ejercicio S5.3: gradientes de la capa de salida ---
W1 = np.random.randn(2,2) * 0.5
b1 = np.zeros(2)
W2 = np.random.randn(2) * 0.5
b2 = 0.0
h, a = forward(x, W1, b1, W2, b2)
dW2, db2 = gradientes(h, a, y)
print(dW2.shape, db2, dW2)
# --- Ejercicio S5.3: comprobación de gradientes ---
x = np.array([0, 1])
y = 1
h, a = forward(x, W1, b1, W2, b2)
dW2, db2 = gradientes(h, a, y)
# --- Ejercicio S5.3: comprobación de gradientes ---
x = np.array([0, 1])
y = 1
h, a = forward(x, W1, b1, W2, b2)
dW2, db2 = gradientes(h, a, y)
print(a, db2, a - y)
# --- Ejercicio S5.4: función de pérdida ---
def perdida(a, y):
    return (a - y) ** 2

# --- Ejercicio S5.4: gradientes de la capa oculta ---
def gradientes_oculta(x, h, a, y, W2):
    delta2 = 2*(a - y) * a * (1-a)
    delta1 = (W2 * delta2) * h * (1-h)
    dW1 = np.outer(x,delta1)
    db1 = delta1
    return dW1, db1

# --- Ejercicio S5.4: comprobar gradientes de la capa oculta ---
np.random.seed(0)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros(2)
W2 = np.random.randn(2) * 0.5
b2 = 0.0
x = np.array([0, 1])
y = 1

h, a = forward(x, W1, b1, W2, b2)
dW1, db1 = gradientes_oculta(x, h, a, y, W2)
print(dW1.shape, db1.shape)
print(dW1)
print(db1)

# comprobación numérica de dW1[1][0]
eps = 1e-5
W1[1, 0] += eps
_, a_mas = forward(x, W1, b1, W2, b2)
W1[1, 0] -= 2 * eps
_, a_menos = forward(x, W1, b1, W2, b2)
W1[1, 0] += eps
numerico = (perdida(a_mas, y) - perdida(a_menos, y)) / (2 * eps)
print(dW1[1, 0], numerico)

# --- Ejercicio S5.5: entrenar la XOR ---
np.random.seed(2)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros(2)
W2 = np.random.randn(2) * 0.5
b2 = 0.0
lr = 1.0
epocas = 20000

for epoca in range(epocas):
    perdida_total = 0
    for x, y in zip(X, Y_xor):
        h, a = forward(x, W1, b1, W2, b2)
        dW2, db2 = gradientes(h, a, y)
        dW1, db1 = gradientes_oculta(x, h, a, y, W2)
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        perdida_total += perdida(a, y)
    if epoca % 2000 == 0:
        print(epoca, perdida_total)

# --- Ejercicio S5.5: comprobar las 4 salidas de la XOR ---
for x, y in zip(X, Y_xor):
    h, a = forward(x, W1, b1, W2, b2)
    print(x, "->", round(float(a), 3), "esperado:", y)