#NUMPY
# ==============================
# %% BLOQUE 1: crear arrays
# ==============================
import numpy as np

a = np.array([5,6,7]) #1 Dimensión
print(a.shape)        #(3,)
print(a.dtype)        #int64 (puede variar en Windows)

# --- Ejercicio 1: shape y dtype ---
#Array 1D del 1 al 10
a_1 = np.array([1,2,3,4,5,6,7,8,9,10])
print(a_1.shape)
print(a_1.dtype)
# --- Ejercicio 2: Matriz de ceros 3x4 ---
# M = np.array([[0,0,0,0],
#              [0,0,0,0],
#              [0,0,0,0]])
#Es más eficiente poner 
M = np.zeros((3,4))
print(M.shape)
print(M.dtype)        #Sale float64 pero se puede cambiar con ((3,4), dtype = int)

# --- Ejercicio 3: Matriz de unos 2x2 ---
M_1 = np.ones((2,2))
print(M_1.shape)
print(M_1.dtype)  

# =====================================
# %% BLOQUE 2: operaciones entre arrays
# =====================================
#Teoría
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b      # [11 22 33]
a * b      # [10 40 90]   (esto NO es producto matricial)
a * 2      # [2 4 6]      (el 2 se aplica a cada elemento)
a ** 2     # [1 4 9]

M = np.ones((3, 4))      # shape (3, 4)
v = np.array([1, 2, 3, 4])   # shape (4,)

print(M + v)    # funciona: v se suma a cada fila de M, resultado (3, 4)

# c = np.array([[1],
#               [2],
#               [3],
#               [4]])            # shape (4, 1)

# v = np.array([10, 20, 30])     # shape (3,)
# print(c + v)
## --- Ejercicio 4: Operaciones básicas con matrices ---
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)                     #[5, 7, 9]
print(a * b)                     #[4, 10, 18]
print(a * 2)                     #[2, 4, 6]
print(a ** 2)                    #[1, 4, 9]
print(np.sum(a * b))             #32

# --- Ejercicio 5: broadcasting (matriz + vector) ---
M = np.arange(12).reshape(3, 4)   # shape (3, 4)
v4 = np.array([1, 2, 3, 4])
v3 = np.array([1, 2, 3])

print(M + v4)                     #shape (3x4), [[1, 3, 5, 7], [5, 7, 9, 11]]
# print(M + v3)                    Error ya que 4 y 3 no son el mismo número
#Arreglamos M + v3 con reshape
v3_col = v3.reshape(3, 1)         #Se podría arreglar también sustituyendo el 1 por un 4
print(v3_col.shape)
print(M + v3_col)

# =====================================
# %% BLOQUE 3: Indexado y slicing
# =====================================
#Teoría
M = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

M[0, 0]    # 0   (primera fila, primera columna)
M[1, 2]    # 6   (segunda fila, tercera columna)
M[-1, -1]  # 11  (índice negativo: cuenta desde el final)
M[0, 1:3]  # [1 2]   (fila 0, columnas 1 y 2)

M[1, :]    # [4 5 6 7]   toda la fila 1
M[:, 2]    # [ 2  6 10]  toda la columna 2

# --- Ejercicio 6: indexado y slicing de una matriz ---
M = np.arange(12).reshape(3, 4)
print(M[1, :])
print(M[:, 2])
print(M[0:2, 0:2])
print(M[2, 2])

# =====================================
# %% BLOQUE 4: Producto matricial
# =====================================
#Teoría
P = np.array([[1, 2],
              [3, 4]])
Q = np.array([[5, 6],
              [7, 8]])

print(P @ Q)
# [[19 22]
#  [43 50]]
#P.T es la traspuesta de la matriz P
# [[1 3]
#  [2 4]]
#Regla de las shapes: (m, n) @ (n, p)  ->  (m, p)
# --- Ejercicio 7: producto matricial y reglas de shape ---
A = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])           # shape (3, 2)
print(A @ B)                     # shape (2, 2), Elemento [0, 0] = 4
#print(A @ A)                      Error debido a que no coinciden las dimensiones ya que 3 != 2
# print(A @ B.T)                   B.shape = (2, 3), por tanto va a dar error ya que 3 =! 2

# --- Ejercicio 8: mi primera neurona ---
x = np.array([0.5, -1, 2])       # entradas
w = np.array([0.2, 0.8, -0.5])   # pesos
b = 0.1                          # sesgo

print(x[0]*w[0] + x[1]*w[1] + x[2]*w[2] + b)
print((x @ w) + b)
# x.shape = (3,)
# 2.shape = (3,)
# x@w no se puede hacer, da error ya que 1 != 3
print((x @ w).shape)             #Da () ya que es un único número
