import numpy as np

# --- Ejercicio 1: sigmoide ---
def sigmoide(z):
    res = 1/(1 + np.exp(-z))
    return res

# Test
# print(sigmoide(0))
# print(sigmoide(np.array([-10, 0, 10])))

# --- Ejercicio 2: forward ---
def forward(x, w, b):
    z = np.dot(w,x) + b
    a = sigmoide(z)
    return a

# Test
print(forward(np.array([1, 1]), np.array([0.0, 0.0]), 0.0))   # debe dar 0.5
print(forward(np.array([1, 1]), np.array([0.5, 0.5]), -0.2))  # debe dar ~0.69

# --- Ejercicio 3: pérdida y gradientes ---
def perdida(a, y):
    res = (a - y)**2
    return res

def gradientes(x, a, y):
    dz = 2*(a - y) * a * (1-a)
    dw = dz * x
    db = dz
    return dw, db

# Test: compara con el gradiente numérico
x = np.array([1, 1])
w = np.array([0.5, 0.5])
b = -0.2
y = 1
h = 1e-5

a = forward(x, w, b)
dw, db = gradientes(x, a, y)
print("analítico:", dw, db)

# gradiente numérico de w[0]:
w_mas = w.copy(); w_mas[0] += h
w_menos = w.copy(); w_menos[0] -= h
dw0_num = (perdida(forward(x, w_mas, b), y) - perdida(forward(x, w_menos, b), y)) / (2*h)

b_mas = b + h
b_menos = b - h
db_num = (perdida(forward(x, w, b_mas), y) - perdida(forward(x, w, b_menos), y))/(2*h)
print("númerico: ",dw0_num, db_num)

# --- Ejercicio 4: entrenar AND ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

w = np.random.randn(2) * 0.1
b = 0.0
lr = 0.5
epocas = 5000
#Bucle de épocas
for epoca in range(epocas):
    perdida_total = 0
    for x,y in zip(X, Y):
        a = forward(x, w, b)
        dw, db = gradientes(x, a, y)
        w = w - lr * dw
        b = b - lr * db
        perdida_total += perdida(a, y)
    if epoca % 500 == 0:
        print(epoca, perdida_total)

# --- Ejercicio S4: comprobar la puerta AND ---
for x, y in zip(X, Y):
    a = forward(x, w, b)
    print(x, "->", round(float(a), 3), "esperado:", y)