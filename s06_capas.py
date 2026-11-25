# --- Ejercicio 1: clase Capa (init + forward) ---
import numpy as np

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

class Capa:
    def __init__(self, n_in, n_out):
        # W: (n_in, n_out), b: (1, n_out)
        self.W = np.random.randn(n_in, n_out) * 1 
        self.b = np.zeros((1, n_out))  

    def forward(self, X):
        self.X = X                    # lo guardamos para el backward
        z = X @ self.W + self.b                       # TODO: X·W + b
        self.a = sigmoide(z)                  # TODO: sigmoide(z)
        return self.a

        # --- Ejercicio 2: Capa.backward ---
    def backward(self, da, lr):
        # da: derivada de la pérdida respecto a la salida de ESTA capa, shape (N, n_out)
        dz = da * self.a * (1-self.a)    # TODO: da * a * (1 - a)
        dW = self.X.T @ dz    # TODO: X traspuesta @ dz   (shape = el de W)
        db = np.sum(dz, axis=0, keepdims=True)   # TODO: suma de dz por filas, manteniendo shape (1, n_out)
        dX = dz @ self.W.T    # TODO: dz @ W traspuesta   (¡calcúlalo ANTES de actualizar W!)

        self.W -= lr * dW
        self.b -= lr * db

        #Visión de dz, dW, db
        # print("dz =", dz)
        # print("dW =", dW)
        # print("db =", db)
        return dX

#Comprobación de forward
# capa = Capa(2,1)
# capa.W = np.array([[0.5], [-0.5]])
# capa.b = np.array([[0.1]])
# print(capa.forward(np.array([[1,0]])))

#Comprobación de backward
capa = Capa(2, 1)
capa.W = np.array([[0.5], [-0.5]])
capa.b = np.array([[0.1]])

X = np.array([[1, 0]])
y = 1

a = capa.forward(X)
da = 2 * (a - y)              # N = 1
dX = capa.backward(da, lr=0)  # lr=0: los pesos no se mueven

print("a  =", a)
print("dX =", dX)

# --- Ejercicio 3: red con lista de capas aprende XOR ---
np.random.seed(5)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

red = [Capa(2, 2), Capa(2, 1)]
lr = 1.0
epocas = 10000

for epoca in range(epocas):
    # 1. Forward: la salida de cada capa es la entrada de la siguiente
    a = X
    for capa in red:
        a = capa.forward(a)                     # TODO

    # 2. Pérdida (MSE)
    perdida = np.mean((a - y) ** 2)

    # 3. Backward: de la última capa a la primera
    da = 2 * (a-y)/X.shape[0]                         # TODO: 2 * (a - y) / N, con N = número de ejemplos
    for capa in reversed(red):
        da = capa.backward(da,lr)                     # TODO: capa.backward(da, lr) devuelve el da de la capa anterior

    if epoca % 1000 == 0:
        print(epoca, perdida)

print(a)   # esperado: ≈ [[0], [1], [1], [0]]