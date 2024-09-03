import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Definir las condiciones de borde y la función para el sistema de ecuaciones
def equations(vars, a, y0, y1, L_target):
    C1, C2, lambd = vars
    
    # Ecuaciones dadas por las condiciones de borde
    eq1 = C1 * np.cosh(-a + C2) - lambd - y0
    eq2 = C1 * np.cosh(a + C2) - lambd - y1
    
    # Ecuación para la longitud de arco
    integral_result, _ = quad(lambda x: np.sqrt(1 + C1**2 * np.sinh(x + C2)**2), -a, a)
    eq3 = integral_result - L_target
    
    return [eq1, eq2, eq3]

# Parámetros iniciales
a = 1.0  # valor de a
y0 = 4.0
y1 = 5.0
L_target = 5.0  # Longitud deseada de la curva
initial_guess = [0, 0, 1]  # Suposiciones iniciales para C1, C2 y lambda

# Resolver el sistema de ecuaciones
C1, C2, lambd = fsolve(equations, initial_guess, args=(a, y0, y1, L_target))

# Graficar la función y(x)
x = np.linspace(-a, a, 100)
y = lambda x: C1 * np.cosh(x + C2) - lambd

plt.plot(x, y(x))
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(f'Grafico de y(x) para para distintos valores de (y0, y1)')
plt.grid(True)
plt.show()

