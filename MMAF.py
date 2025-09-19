# En este archivo estará la librería de funciones que se utilizarán en el programa principal

#LIBRERIAS: 
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider, interactive_output, VBox, FloatText, HBox, Text
import math

# Función Actividad 1 (Población de Ardillas)
# -------------------------------------------------------------------------------------
def calcular_poblacion(juveniles_iniciales, adultas_iniciales, tasa_reproduccion, tasa_conversion, tasa_supervivencia, epocas):
    """
    Calcula la población de ardillas a lo largo de varios años.
    
    Parámetros:
        juveniles_iniciales (int): Número inicial de ardillas juveniles.
        adultas_iniciales (int): Número inicial de ardillas adultas.
        tasa_reproduccion (float): Número promedio de juveniles generados por cada adulta.
        tasa_conversion (float): Proporción de juveniles que pasan a ser adultas.
        tasa_supervivencia (float): Proporción de adultas que sobreviven.
        anos (int): Número de años a calcular.
        
    Retorna:
        lista_poblaciones (list): Lista con las poblaciones totales de cada año.
    """
    # Valores iniciales
    juveniles = juveniles_iniciales
    adultas = adultas_iniciales

    lista_poblaciones = []

    for epocas in range(1, epocas + 1):
        # Cálculos
        juveniles_nuevas = adultas * tasa_reproduccion
        juveniles_que_pasan = juveniles * tasa_conversion
        adultas_sobrevivientes = adultas * tasa_supervivencia

        # Nuevos valores de juveniles y adultas
        juveniles = juveniles_nuevas
        adultas = juveniles_que_pasan + adultas_sobrevivientes

        # Población total
        poblacion_total = juveniles + adultas
        lista_poblaciones.append((epocas, juveniles, adultas, poblacion_total))
        # --- RESULTADOS ---
    print(f"{'Año':<5}{'Juveniles':<15}{'Adultas':<15}{'Población Total':<15}")
    for epocas, juveniles, adultas, total in lista_poblaciones:
        print(f"{epocas:<5}{int(juveniles):<15}{int(adultas):<15}{int(total):<15}")
    
    return 
# -------------------------------------------------------------------------------------

# Función Actividad 2 (Diferencia de cuadrados)
# -------------------------------------------------------------------------------------
def analizar_diferencia_cuadrados(radio_exterior, radio_interior, altura): 
    """
    Analiza el impacto de errores comunes al calcular el volumen de una capa cilíndrica.
    Comparación entre:
    - Cálculo correcto: π(R^2 - r^2)h
    - Cálculo erróneo: π(R - r)^2h
    """
    # Cálculo correcto
    volumen_correcto = math.pi * (radio_exterior**2 - radio_interior**2) * altura

    # Error común: cuadrado de la diferencia
    volumen_erroneo = math.pi * (radio_exterior - radio_interior)**2 * altura

    # Diferencia entre ambos cálculos
    diferencia = volumen_correcto - volumen_erroneo

    # Resultados
    print("=== Resultados ===")
    print(f"Radio exterior (R): {radio_exterior} m")
    print(f"Radio interior (r): {radio_interior} m")
    print(f"Altura (h): {altura} m")
    print("\n--- Cálculo correcto ---")
    print(f"Volumen correcto (π(R² - r²)h): {volumen_correcto:.2f} m³")
    print("\n--- Cálculo erróneo ---")
    print(f"Volumen erróneo (π(R - r)²h): {volumen_erroneo:.2f} m³")
    print("\n--- Diferencia entre ambos ---")
    print(f"Diferencia: {diferencia:.2f} m³")
    return 
# -------------------------------------------------------------------------------------
def analizar_diferencia_cubo(radio_exterior, radio_interior): 
    """
    Analiza el impacto de errores comunes al calcular el volumen de una capa cilíndrica.
    Comparación entre:
    - Cálculo correcto: 4/3π(R^3 - r^3)
    - Cálculo erróneo: 4/3π(R - r)^3
    """
    # Cálculo correcto
    volumen_correcto = (4/3) * math.pi * (radio_exterior**3 - radio_interior**3)

    # Error común: cuadrado de la diferencia
    volumen_erroneo = (4/3) * math.pi * (radio_exterior - radio_interior)**3

    # Diferencia entre ambos cálculos
    diferencia = volumen_correcto - volumen_erroneo

    # Resultados
    print("=== Resultados ===")
    print(f"Radio exterior (R): {radio_exterior} m")
    print(f"Radio interior (r): {radio_interior} m")
    print("\n--- Cálculo correcto ---")
    print(f"Volumen correcto (4/3π(R³ - r³)): {volumen_correcto:.2f} m³")
    print("\n--- Cálculo erróneo ---")
    print(f"Volumen erróneo (4/3π(R - r)³): {volumen_erroneo:.2f} m³")
    print("\n--- Diferencia entre ambos ---")
    print(f"Diferencia: {diferencia:.2f} m³")
    return 
# Funciones Actividad 3 (Pepito Perez)
# -------------------------------------------------------------------------------------
# Función para obtener los costos asociados a las tres propuestas
def obtener_costos(t_seleccionado):
    # Definir las funciones de costo
    valor_interprase = 950000 + 5500 * t_seleccionado + 30000 + 25 * t_seleccionado
    valor_soluciones = 1100000 + 150 * t_seleccionado + 50000 - 10 * t_seleccionado
    valor_pepito     = -2000 * t_seleccionado**2 + 20000 * t_seleccionado + 1130000
    
    return valor_interprase, valor_soluciones, valor_pepito

# Función para mostrar los costos asociados a las tres propuestas
def mostrar_costos(t_seleccionado):
    # Obtener los costos
    valor_interprase, valor_soluciones, valor_pepito = obtener_costos(t_seleccionado)
    
    # Imprimir los valores en la consola
    print(f"Para t = {t_seleccionado}:")
    print(f"  - Interprase         : ${valor_interprase:,.2f}")
    print(f"  - Soluciones Express : ${valor_soluciones:,.2f}")
    print(f"  - Pepito             : ${valor_pepito:,.2f}")

    return None


# Función para el slider
def interactuar_con_slider():
    return interact(
        mostrar_costos, 
        t_seleccionado=FloatSlider(
            value=0,      # Valor inicial del slider
            min=0,        # Valor mínimo
            max=20,       # Valor máximo
            step=0.5,     # Incremento
            description='Tiempo (t)'  # Etiqueta del slider
        )
    )
# -------------------------------------------------------------------------------------

# Funciones auxiliares para Taller 3 y Python 3
# -------------------------------------------------------------------------------------
# Función para calcular el discriminante
def calcular_discriminante(a, b, c):
  discriminante = b**2 - 4*a*c
  print("discriminante: ", discriminante)
  return discriminante

# Función para resolver ecuaciones cuadráticas
def resolver_ecuacion_cuadratica(a, b, c):
  discriminante = calcular_discriminante(a,b,c)
  #discriminante = b*2 - 4 a* c
  x1 = (-b + math.sqrt(discriminante)) / (2*a)
  x2 = (-b - math.sqrt(discriminante)) / (2*a)

  print("las soluciones son: ", x1, "y", x2)
  return x1, x2
#--------------------------------------------------------------------------------------

# Funciones Actividad 4 (Brecha de acceso a internet Buenaventura)
# -------------------------------------------------------------------------------------
# Función para calcular la temperatura en función del flujo de datos D
def calcular_temperatura(B):
    D = np.linspace(0, 10, 400)
    T = 6 * D + B
    return D, T

# Función para mostrar la gráfica de la temperatura y el flujo crítico
def mostrar_temperatura(B):
    D, T = calcular_temperatura(B)
    
    plt.figure(figsize=(8, 4))
    plt.plot(D, T, 'm-', lw=2, label='T(D) = 6D + B')
    plt.axhline(50, color='k', linestyle='--', label='T_c = 60°C')
    plt.xlabel('Flujo de datos D')
    plt.ylabel('Temperatura T (°C)')
    plt.title('Función de Temperatura del Cable')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 101, 10))
    plt.show()

    # Cálculo del flujo crítico D_c
    D_c = (60 - B) / 6
    print(f'Para B = {B:.2f}, el flujo crítico es D_c = {D_c:.2f}')

# Función para calcular el desgaste del cable en función del tiempo t
def calcular_desgaste():
    t_vals = np.linspace(0, 20, 400)
    d = -4/5 * t_vals + 12
    return t_vals, d

# Función para mostrar la gráfica del desgaste del cable
def mostrar_desgaste(D_c):
    t_vals, d = calcular_desgaste()
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, d, 'b-', lw=2, label='d(t) = -4/5t + 12')
    plt.axhline(D_c, color='r', linestyle='--', label=f'D_c = {D_c:.2f}')
    plt.xlabel('Tiempo t (años)')
    plt.ylabel('Capacidad D(t)')
    plt.title('Función de Capacidad máxima de transmisión del Cable')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 15, 1))
    plt.show()

# Función para la interacción con el slider de temperatura
def interactuar_con_temperatura_2():
    return interact(
        mostrar_temperatura, 
        B=FloatSlider(value=10, min=0, max=20, step=0.5, description='Parámetro B')
    )

# Función para la interacción con el slider de desgaste
def interactuar_con_desgaste_2():
    return interact(
        mostrar_desgaste, 
        D_c=FloatSlider(value=5, min=0, max=20, step=0.1, description='Capacidad Crítica D_c')
    )

# Funciones auxiliares para Python 4
# -------------------------------------------------------------------------------------

# Grafica la recta que pasa por dos puntos en el plano cartesiano.
def grafica_recta(punto1, punto2):
  x1, y1 = punto1
  x2, y2 = punto2

  # Verificar si es una recta vertical
  if x1 == x2:
    m = None
    b = None
    print(f"Ecuación de la recta: x = {x1}")
    x_vals = np.full(200, x1)
    y_vals = np.linspace(min(y1, y2)-5, max(y1, y2)+5, 200)
  else:
    # calcular pendiente y ordenada
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    print(f"Ecuación de la recta: y = {m:.2f}x + {b:.2f}")
    x_vals = np.linspace(min(x1, x2)-5, max(x1, x2)+5, 200)
    y_vals = m*x_vals + b

  # graficar
  plt.figure(figsize=(6,4))
  plt.plot(x_vals, y_vals, label="Recta")
  plt.scatter([x1, x2], [y1, y2], color="red", zorder=5, label="Puntos dados")
  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.grid(True)
  plt.legend()
  plt.title("Recta a partir de dos puntos")
  plt.show()

# Grafica la recta a partir de su pendiente y un punto dado.
def grafica_recta_pendiente_punto(m, punto):
  x0, y0 = punto

  # calcular ordenada al origen: y = mx + b -> b = y0 - m*x0
  b = y0 - m*x0
  print(f"Ecuación de la recta: y = {m:.2f}x + {b:.2f}")

  # generar valores de x y la correspondiente y
  x_vals = np.linspace(x0 - 5, x0 + 5, 200)
  y_vals = m*x_vals + b

  # graficar
  plt.figure(figsize=(6,4))
  plt.plot(x_vals, y_vals, label=f"Recta: y={m:.2f}x+{b:.2f}")
  plt.scatter([x0], [y0], color="red", zorder=5, label="Punto dado")
  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.grid(True)
  plt.legend()
  plt.title("Recta a partir de pendiente y un punto")
  plt.show()

# Grafica dos rectas a partir de sus pendientes y puntos dados.
def grafica_dos_rectas(m1, punto1, m2, punto2):
  x1, y1 = punto1
  x2, y2 = punto2

  # calcular ordenadas al origen
  b1 = y1 - m1*x1
  b2 = y2 - m2*x2

  print(f"Recta 1: y = {m1:.2f}x + {b1:.2f}")
  print(f"Recta 2: y = {m2:.2f}x + {b2:.2f}")

  # generar valores de x
  x_vals = np.linspace(min(x1, x2) - 5, max(x1, x2) + 5, 200)

  # calcular valores de y
  y_vals1 = m1*x_vals + b1
  y_vals2 = m2*x_vals + b2

  # graficar
  plt.figure(figsize=(6,4))
  plt.plot(x_vals, y_vals1, label=f"Recta 1: y={m1:.2f}x+{b1:.2f}")
  plt.plot(x_vals, y_vals2, label=f"Recta 2: y={m2:.2f}x+{b2:.2f}")
  plt.scatter([x1, x2], [y1, y2], color="red", zorder=5, label="Puntos dados")
  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.grid(True)
  plt.legend()
  plt.title("Gráfica de dos rectas a partir de pendientes y puntos")
  plt.show()

# Función para graficar una parábola que pasa por tres puntos dados
def grafica_parabola_puntos(p1, p2, p3):
  # Construir el sistema lineal para resolver a, b, c de y = ax^2 + bx + c
  A = np.array([
      [p1[0]**2, p1[0], 1],
      [p2[0]**2, p2[0], 1],
      [p3[0]**2, p3[0], 1]
  ])
  Y = np.array([p1[1], p2[1], p3[1]])

  # Resolver el sistema
  a, b, c = np.linalg.solve(A, Y)

  print("La parábola obtenida es: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))

  # Graficar la parábola
  x_vals = np.linspace(min(p1[0], p2[0], p3[0]) - 1, max(p1[0], p2[0], p3[0]) + 1, 200)
  y_vals = a*x_vals**2 + b*x_vals + c

  plt.figure(figsize=(6,4))
  plt.plot(x_vals, y_vals, label="Parábola ajustada")
  plt.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color="red", zorder=5, label="Puntos dados")
  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.legend()
  plt.title("Parábola que pasa por tres puntos")
  plt.grid(True)
  plt.show()

  return a, b, c


# Función para graficar una parábola en forma canónica: y = a(x-h)^2 + k
def grafica_parabola_can(a, h, k):
    # Definir el rango de valores para x
  x_vals = np.linspace(h-5, h+5, 200)
  y_vals = a*(x_vals - h)**2 + k

  print("La parábola en forma canónica es: y = {}(x - {})^2 + {}".format(a, h, k))

  # Graficar
  plt.figure(figsize=(6,4))
  plt.plot(x_vals, y_vals, label="Parábola canónica")
  plt.scatter([h], [k], color="red", zorder=5, label="Vértice ({},{})".format(h,k))
  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.legend()
  plt.title("Parábola en forma canónica")
  plt.grid(True)
  plt.show()

  return None
    
# Función para mostrar una parábola en forma canónica interactiva
def interactive_parabola_can():
  """
  Crea un widget interactivo para graficar una parábola en forma canónica:
      y = a(x-h)^2 + k

  Los parámetros a, h, k se controlan con sliders.
  """
  
  # Sliders para manipular los parámetros de la parábola
  a_slider = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='a')
  h_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.5, description='h')
  k_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.5, description='k')

  # Función que actualiza la gráfica
  def update_grafica(a, h, k):
    x_vals = np.linspace(h-5, h+5, 200)
    y_vals = a*(x_vals - h)**2 + k

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, y_vals, label=f"y = {a:.2f}(x - {h:.2f})^2 + {k:.2f}")
    plt.scatter([h], [k], color="red", label=f"Vértice ({h},{k})")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Parábola en forma canónica (interactiva)")
    plt.show()

  # Crear widget interactivo
  parabola_widget = widgets.interactive(update_grafica, a=a_slider, h=h_slider, k=k_slider)
  display(parabola_widget)


# Función para mostrar una parábola en forma ccuadrática interactiva
def interactive_cuadratica():
  """
  Crea un widget interactivo para graficar una función cuadrática en forma general:
      y = ax^2 + bx + c

  Los parámetros a, b, c se controlan con sliders.
  """

  # Sliders para los parámetros
  a_slider = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='a')
  b_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.5, description='b')
  c_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.5, description='c')

  # Función que actualiza la gráfica
  def update_grafica_cuad(a, b, c):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = a*x_vals**2 + b*x_vals + c

    # vértice
    if a != 0:
      h = -b / (2*a)
      k = a*h**2 + b*h + c
    else:
      h, k = None, None

    plt.figure(figsize=(6,4))
    plt.plot(x_vals, y_vals, label=f"y = {a:.2f}x² + {b:.2f}x + {c:.2f}")
    if h is not None:
      plt.scatter([h], [k], color="red", label=f"Vértice ({h:.2f},{k:.2f})")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Función cuadrática (interactiva)")
    plt.show()

  # Crear widget interactivo
  cuadratica_widget = widgets.interactive(update_grafica_cuad, a=a_slider, b=b_slider, c=c_slider)
  display(cuadratica_widget)

#--------------------------------------------------------------------------------------

# Funciones Actividad 5 (Deflexión de una viga en voladizo)
# -------------------------------------------------------------------------------------
def deflexion_viga(x, t):
    """
    Calcula la deflexión de la viga para un nivel de carga proporcional t,
    usando la función:
        d(x) = -t * (1/16000) * (60*x**2 - x**3)
    """
    return -t * (1/16000) * (60*x**2 - x**3)

# Función para mostrar la animación de la deflexión de la viga
def animar_deflexion_viga():
    x = np.linspace(0, 20, 400)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 20)
    y_min = deflexion_viga(20, 1)
    ax.set_ylim(y_min * 1.2, 0)

    line, = ax.plot([], [], lw=3, color='blue')

    def init():
        line.set_data([], [])
        ax.set_xlabel("Posición a lo largo de la viga (m)")
        ax.set_ylabel("Deflexión (cm)")
        ax.set_title("Animación de deflexión en viga en voladizo")
        return line,

    def update(frame):
        t = frame  # t varía de 0 a 1 (0% a 100% de carga)
        y = deflexion_viga(x, t)
        line.set_data(x, y)
        ax.set_title(f"Deflexión de la viga (Carga aplicada: {t*100:.1f}%)")
        return line,

    frames = np.linspace(0, 1, 100)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=True)

    anim.save("viga_animation.gif", writer="pillow", fps=30)

# Funciones para encontrar el intervalo seguro 
# --------------------------------------------------------------------------------------------
# Función de deflexión: d(x, C_g) en cm, x en m
def d(x, C_g):
    return (C_g / 16000) * (60 * x**2 - x**3)

def encontrar_intersecciones(d_objetivo, C_g):
    x_dom = np.linspace(0, 20, 400)
    dif = d(x_dom, C_g) - d_objetivo
    indices = np.where(np.diff(np.sign(dif)))[0]
    soluciones = []
    for i in indices:
        sol = fsolve(lambda x: d(x, C_g) - d_objetivo, x_dom[i])
        soluciones.append(sol[0])
    return sorted(soluciones)

def interactuar(C_g):  
    d_lim = 0.1  
    intersecciones = encontrar_intersecciones(d_lim, C_g)
    x = np.linspace(0, 20, 400)
    y = d(x, C_g)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=r'$d(x,C_g)=\frac{C_g}{16000}(60x^2-x^3)$')
    plt.axhline(d_lim, color='red', linestyle='--', label='dlim = 0.1 cm')

    if intersecciones:
        for x_int in intersecciones:
            y_int = d(x_int, C_g)
            plt.axvline(x_int, color='green', linestyle='--', label=f'x = {x_int:.3f} m')
            plt.scatter(x_int, y_int, color='blue', s=100, zorder=5)
    else:
        print("No se encontraron intersecciones en el dominio analizado.")

    plt.xlabel('Posición a lo largo de la viga (m)')
    plt.ylabel('Deflexión (cm)')
    plt.title('Flexión de la viga en volaanalizar_diferencia_de_cubosdizo')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)
    plt.show()

    print(f'Para d_objetivo = {d_lim:.3f} cm y C_g = {C_g:.2f}.')
    if intersecciones:
        print('Intersección en x = ' + ", ".join(f"{xi:.3f}" for xi in intersecciones) + " m.")

def visualizar_limites():
    C_g_slider = FloatSlider(min=0, max=1, step=0.05, value=1.0, description='C_g')
    out = interactive_output(interactuar, {'C_g': C_g_slider})
    ui = VBox([C_g_slider])
    display(ui, out)


# Funciones Actividad 6 (Crecimiento Celulas Cancerigenas)
# ------------------------------------------------------------------------------------
def modelo_exponencial(t, N0, doubling_time):
    # Calcula el crecimiento exponencial.
    return N0 * 2**(t/doubling_time)

def modelo_logistico(t, K, A, r):
    # Calcula el crecimiento logístico.
    return K / (1 + A * np.exp(-r * t))

def plot_growth(Td=3, N0=100, K=1000, A=9, r=0.2, Tf=30):
    # Genera la gráfica del crecimiento celular con y sin tratamiento.
    t = np.linspace(0, Tf, 10*Tf)
    N_exponencial = modelo_exponencial(t, N0, Td)
    N_logistico = modelo_logistico(t, K, A, r)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, N_exponencial, label="Sin tratamiento (Exponencial)")
    ax.plot(t, N_logistico, label="Con tratamiento (Logístico)")
    ax.axhline(K/2, color='gray', linestyle='--', label="Carga máxima/2")
    
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Número de células")
    ax.set_title("Simulación del Crecimiento de Células Cancerígenas")
    ax.legend()
    ax.set_xlim(0, Tf + 2)
    ax.set_ylim(0, K + 200)
    ax.grid(True)
    
    plt.show()

def visualizar_crecimiento_cancer():
    # Configura los controles y permite visualizar la simulación del crecimiento celular.
    Td_slider = FloatSlider(value=3, min=1, max=10, step=0.5, description="Td:")
    N0_slider = IntSlider(value=100, min=50, max=500, step=10, description="N0:")
    K_slider = IntSlider(value=1000, min= 500, max=2000, step=100, description="K:")
    A_slider = FloatSlider(value=9, min=1, max=20, step=1, description="A:")
    r_slider = FloatSlider(value=0.2, min=0.05, max=1.0, step=0.05, description="r:")
    Tf_slider = IntSlider(value=30, min=10, max=60, step=5, description="Tf:")
    
    ui = VBox([Td_slider, N0_slider, K_slider, A_slider, r_slider, Tf_slider])
    out = interactive_output(plot_growth, {
        'Td': Td_slider, 'N0': N0_slider, 'K': K_slider,
        'A': A_slider, 'r': r_slider, 'Tf': Tf_slider
    })
    
    display(ui, out)


# Funciones Actividad 7 (Modelo de conexiones satelitales)
# ------------------------------------------------------------------------------------
def push_radially(pt, factor=1.22):
    # escala radial
    return np.array(pt) * factor

def surface_coverage_distance(h):
    # distancia superficial lineal
    R = 6378  # km
    return np.sqrt((R + h)**2 - R**2)

def required_threshold(D, m):
    # umbral requerido
    return D/2 + m

def view_angle(h):
    # ángulo de visión
    R = 6378
    return np.arccos(R / (R + h))

def plot_satellite(D, h, m):
    # dibuja Tierra, ciudades y cobertura
    R = 6378
    theta0 = 7.6 * np.pi / 180
    d = surface_coverage_distance(h)
    th_view = view_angle(h)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # tierra
    ax.add_patch(plt.Circle((0, 0), R, color='lightblue', alpha=0.4))

    # ciudades
    half = theta0 / 2
    Q = [R * np.cos(half), R * np.sin(half)]
    M = [R * np.cos(-half), R * np.sin(-half)]
    ax.plot(*Q, 'ko'); ax.text(*push_radially(Q), 'Quibdó')
    ax.plot(*M, 'ko'); ax.text(*push_radially(M), 'Mitú')

    # satélite
    sat = [R + h, 0]
    ax.plot(*sat, 'ro'); ax.text(*push_radially(sat, 1.03), 'SAT')

    # altura
    ax.text(0.98, 0.95, f'h={h:.0f} km', transform=ax.transAxes,
            ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))

    # cobertura
    phi = np.linspace(-th_view, th_view, 400)
    ax.plot(R * np.cos(phi), R * np.sin(phi), 'r--')

    # límites
    mrg = 2200
    ax.set_xlim(-R - mrg, R + mrg); ax.set_ylim(-R - mrg, R + mrg)
    ax.set_xlabel('km'); ax.set_ylabel('km')
    ax.set_title(f'd={d:.0f} km (req ≥ {required_threshold(D, m):.0f})')
    ax.grid(True)
    plt.show()

def create_widgets():
    # widgets
    R = 6378
    theta0 = 7.6 * np.pi / 180
    D_pred = R * theta0
    D_text = FloatText(value=np.round(D_pred, 2), description='D (km):')
    h_slider = FloatSlider(value=500, min=100, max=1200, step=10, description='h (km):')
    m_slider = FloatSlider(value=0, min=0, max=300, step=10, description='margen (km):')
    return D_text, h_slider, m_slider

def build_interface_satelite():
    # interfaz interactiva
    D_text, h_slider, m_slider = create_widgets()
    ui = VBox([HBox([D_text]), h_slider, m_slider])
    out = interactive_output(plot_satellite,
                             {'D': D_text, 'h': h_slider, 'm': m_slider})
    display(ui, out)

# -------------------------------------------------------------------------------------
# Funciones Herramientas Computacionales 7
# -------------------------------------------------------------------------------------

# Definir la función seno con desfase y la función seno original
def plot_sine(desfase=0):
    # Dominio de 0 a 2π para un ciclo completo
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x + desfase)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=2)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.ylim(-1.5, 1.5)
    plt.title(fr'$y = \sin(x + {desfase:.2f})$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.grid(True)
    plt.show()

# Función para graficar la onda seno con amplitud, frecuencia, desfase y término constante
def graficar_seno(amplitud, frecuencia, desfase, constante):
    # Definir el rango de los valores de x
    x = np.linspace(-3*np.pi, 3 * np.pi, 1000)  # De 0 a 4π para ver varios ciclos

    # Calcular los valores de y usando la función seno
    y = amplitud * np.sin(frecuencia * x + desfase) + constante

    # Limpiar la figura
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"$y = {amplitud} \\cdot \\sin({frecuencia}x + {desfase}) + {constante}$", color='blue')
    plt.title(f"Onda Senoidal: Amplitud = {amplitud}, Frecuencia = {frecuencia}, Desfase = {desfase}, Constante = {constante}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-3*np.pi,3*np.pi)
    plt.xticks([-3 * np.pi,-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi,3 * np.pi], 
               [r'$-3\pi$',r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$',r'$3\pi$'])
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()

# Función para graficar la onda coseno con amplitud, frecuencia, desfase y término constante
def graficar_coseno(amplitud, frecuencia, desfase, constante):
    # Definir el rango de los valores de x
    x = np.linspace(-3*np.pi, 3 * np.pi, 1000)  # De 0 a 4π para ver varios ciclos

    # Calcular los valores de y usando la función coseno
    y = amplitud * np.cos(frecuencia * x + desfase) + constante

    # Limpiar la figura
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"$y = {amplitud} \\cdot \\cos({frecuencia}x + {desfase}) + {constante}$", color='green')
    plt.title(f"Onda Cosenoidal: Amplitud = {amplitud}, Frecuencia = {frecuencia}, Desfase = {desfase}, Constante = {constante}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-3*np.pi,3*np.pi)
    plt.xticks([-3 * np.pi,-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi,3 * np.pi], 
               [r'$-3\pi$',r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$',r'$3\pi$'])
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


# Función para graficar la onda tangente con amplitud, frecuencia, desfase y término constante
def graficar_tangente(amplitud, frecuencia, desfase, constante):
    # Definir el rango de los valores de x, evitando las asíntotas de la tangente
    x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)  # De -2π a 2π para ver varios ciclos

    # Calcular los valores de y usando la función tangente
    y = amplitud * np.tan(frecuencia * x + desfase) + constante

    # Limitar los valores de y para evitar que crezcan demasiado cerca de las asíntotas
    y = np.clip(y, -10, 10)  # Limitar el rango de y entre -10 y 10

    # Limpiar la figura
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"$y = {amplitud} \\cdot \\tan({frecuencia}x + {desfase}) + {constante}$", color='purple')
    plt.title(f"Onda Tangencial: Amplitud = {amplitud}, Frecuencia = {frecuencia}, Desfase = {desfase}, Constante = {constante}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-3*np.pi,3*np.pi)
    plt.xticks([-3 * np.pi,-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi,3 * np.pi], 
               [r'$-3\pi$',r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$',r'$3\pi$'])
    plt.grid(True)

    # Dibujar las asíntotas verticales en múltiplos de π/frecuencia
    for i in range(-2, 3):
        plt.axvline(x=(i * np.pi - desfase) / frecuencia, color='red', linestyle='--', alpha=0.5)

    plt.legend(loc="upper right")
    plt.show()

# Función para graficar la onda seno con desfase, con el coseno fijo de fondo
def graficar_seno_desfase(desfase):
    # Definir el rango de los valores de x
    x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)  # De -2π a 2π para ver varios ciclos

    # Calcular los valores de y para la función seno y coseno
    y_seno = np.sin(x + desfase)  # Función seno con desfase
    y_cos = np.cos(x)             # Función coseno fija

    # Limpiar la figura
    plt.figure(figsize=(10, 5))

    # Graficar la función coseno como fondo
    plt.plot(x, y_cos, label=f"$\\cos(x)$ (fija)", color='blue', linestyle='--', alpha=0.7)

    # Graficar la función seno con el desfase ajustado
    plt.plot(x, y_seno, label=f"$\\sin(x + {desfase:.2f})$", color='purple')

    # Ajustar título y etiquetas
    plt.title(f"Función Seno con Desfase: Desfase = {desfase:.2f} rad")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-3*np.pi,3*np.pi)
    plt.xticks([-3 * np.pi,-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi,3 * np.pi], 
               [r'$-3\pi$',r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$',r'$3\pi$'])
    plt.grid(True)

    plt.legend(loc="upper right")
    plt.show()

# Función para graficar las funciones trigonométricas inversas con una cuadrícula cuadrada
def graficar_inversas():
    # Definir el rango de x para las funciones inversas
    x_sin_cos = np.linspace(-1, 1, 1000)
    x_tan = np.linspace(-10, 10, 1000)

    # Calcular las funciones inversas
    y_arcsin = np.arcsin(x_sin_cos)
    y_arccos = np.arccos(x_sin_cos)
    y_arctan = np.arctan(x_tan)

    # Crear la figura con tres subgráficos (uno para cada función)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    # Graficar la función arcsin(x)
    ax1.plot(x_sin_cos, y_arcsin, label=r"$\arcsin(x)$", color='blue', linewidth=2)
    ax1.set_title("Función Inversa del Seno", fontsize=14)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.grid(True)
    ax1.set_aspect('equal')  # Cuadrícula cuadrada
    ax1.legend(fontsize=12)
    
    # Etiquetas del eje y para arcsin
    ax1.set_yticks([-np.pi/2, 0, np.pi/2])
    ax1.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"])

    # Graficar la función arccos(x)
    ax2.plot(x_sin_cos, y_arccos, label=r"$\arccos(x)$", color='green', linewidth=2)
    ax2.set_title("Función Inversa del Coseno", fontsize=14)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.grid(True)
    ax2.set_aspect('equal')  # Cuadrícula cuadrada
    ax2.legend(fontsize=12)
    
    # Etiquetas del eje y para arccos
    ax2.set_yticks([0, np.pi/2, np.pi])
    ax2.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])

    # Graficar la función arctan(x)
    ax3.plot(x_tan, y_arctan, label=r"$\arctan(x)$", color='purple', linewidth=2)
    ax3.set_title("Función Inversa de la Tangente", fontsize=14)
    ax3.set_xlabel("x", fontsize=12)
    ax3.set_ylabel("y", fontsize=12)
    ax3.grid(True)
    ax3.set_aspect('equal')  # Cuadrícula cuadrada
    ax3.legend(fontsize=12)

    # Etiquetas del eje y para arctan
    ax3.set_yticks([-np.pi/2, 0, np.pi/2])
    ax3.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"])

    # Ajustar el espaciado y mostrar las gráficas
    plt.tight_layout()
    plt.show()

# Función para graficar y resolver la ecuación seno
def resolver_seno(A):
    # Definir el rango de x (el ángulo θ) en el intervalo [-2π, 2π]
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    
    # Calcular la función seno
    y = np.sin(x)
    
    # Graficar la función seno
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=r"$\sin(\theta)$", color='blue')
    
    # Graficar la línea horizontal en y = A
    plt.axhline(y=A, color='red', linestyle='--', label=f"$y = {A}$")
    
    # Encontrar las soluciones gráficamente
    sol_x = x[np.isclose(y, A, atol=0.01)]
    
    # Graficar las soluciones encontradas
    for sol in sol_x:
        plt.plot(sol, A, 'ro')  # Poner un punto en cada solución
        plt.text(sol, A + 0.05, f"$\\theta = {sol:.2f}$", fontsize=12, color='red')
    
    # Ajustar el gráfico
    plt.title(f"Soluciones de $\\sin(\\theta) = {A}$ en el intervalo $[-2\\pi, 2\\pi]$")
    plt.xlabel("$\\theta$")
    plt.ylabel("$\\sin(\\theta)$")
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-1.5, 1.5)
    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.grid(True)
    plt.legend()
    plt.show()

# Función para graficar la ecuación trigonométrica
def graficar_funcion_trigonometrica(rango_x=(0, 2 * np.pi), num_puntos=1000, figsize=(8, 6)):
    """
    Grafica la ecuación 2·sin²(x) − sin(x) − 1 sobre el rango dado y marca
    los valores x = π/2, 7π/6 y 11π/6.

    Parámetros:
      - rango_x (tuple): Tupla (x_min, x_max) para el dominio.
      - num_puntos (int): Cantidad de puntos para el muestreo en x.
      - figsize (tuple): Tamaño de la figura (ancho, alto).
    """
    # Definir el rango de x
    x = np.linspace(rango_x[0], rango_x[1], num_puntos)
    # Evaluar la función
    y = 2 * np.sin(x)**2 - np.sin(x) - 1

    # Crear la figura
    plt.figure(figsize=figsize)
    plt.plot(x, y, label=r"$2\sin^2(x) - \sin(x) - 1$")

    # Líneas auxiliares
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(np.pi / 2, color='red', linestyle='--', label=r"$\frac{\pi}{2}$")
    plt.axvline(7 * np.pi / 6, color='red', linestyle='--', label=r"$\frac{7\pi}{6}$")
    plt.axvline(11 * np.pi / 6, color='red', linestyle='--', label=r"$\frac{11\pi}{6}$")

    # Título y etiquetas (cadena cruda para evitar invalid escape)
    plt.title(r"Gráfica de $2\sin^2(x) - \sin(x) - 1 = 0$", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)

    plt.grid(True)
    plt.legend()
    plt.show()

# Función para graficar las funciones trigonométricas secante, cosecante y cotangente

def graficar_funciones_trigonometricasotras():
    # Definir el rango de x, excluyendo puntos donde sin(x) y cos(x) son 0
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    # Definir las funciones secante, cosecante y tangente
    def sec(x):
        return 1 / np.cos(x)

    def csc(x):
        return 1 / np.sin(x)

    def cot(x):
        return 1/np.tan(x)

    # Crear una figura con tres gráficos separados
    plt.figure(figsize=(10, 12))

    # Gráfico de sec(x)
    plt.subplot(3, 1, 1)  # Primer subplot
    plt.plot(x, sec(x), label=r"$\sec(x)$", color='blue', linewidth=2)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(x=0, color='black',linewidth=1)
    plt.ylim(-10, 10)
    plt.title(r"Gráfica de $\sec(x)$", fontsize=16)
    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.grid(True)
    plt.legend()

    # Gráfico de csc(x)
    plt.subplot(3, 1, 2)  # Segundo subplot
    plt.plot(x, csc(x), label=r"$\csc(x)$", color='green', linewidth=2)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(x=0, color='black',linewidth=1)
    plt.ylim(-10, 10)
    plt.title(r"Gráfica de $\csc(x)$", fontsize=16)
    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.grid(True)
    plt.legend()

    # Gráfico de cotan(x)
    plt.subplot(3, 1, 3)  # Tercer subplot
    plt.plot(x, cot(x), label=r"$\cot(x)$", color='purple', linewidth=2)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    plt.ylim(-10, 10)
    plt.title(r"Gráfica de $\cot(x)$", fontsize=16)
    plt.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               [r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.grid(True)
    plt.legend()

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()
# Función para visualizar la identidad pitagórica

def identidad_pitagorica(theta_deg):
    # Convertir el ángulo a radianes
    theta_rad = np.deg2rad(theta_deg)

    # Calcular el seno y el coseno
    seno = np.sin(theta_rad)
    coseno = np.cos(theta_rad)
    identidad = seno**2 + coseno**2

    # Limpiar la figura
    plt.figure(figsize=(6, 6))

    # Dibujar el círculo unitario
    circle = plt.Circle((0, 0), 1, color='lightblue', fill=False)
    plt.gca().add_patch(circle)

    # Dibujar el triángulo
    plt.plot([0, coseno], [0, seno], 'k-', lw=2)
    plt.plot([coseno, coseno], [0, seno], 'k--')
    plt.plot([0, coseno], [0, 0], 'k--')

    # Anotar los valores de seno, coseno y la identidad
    plt.text(0.1, 0.9, f"$\\sin^2(\\theta) + \\cos^2(\\theta) = {identidad:.2f}$", fontsize=12, color='red')
    plt.text(0.5, seno / 2, f"$\\sin(\\theta) = {seno:.2f}$", fontsize=12, color='blue')
    plt.text(coseno / 2, -0.1, f"$\\cos(\\theta) = {coseno:.2f}$", fontsize=12, color='green')

    # Ajustes de la gráfica
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.title(f"Identidad Pitagórica para $\\theta = {theta_deg}^\\circ$")
    plt.show()

# Función para dibujar los dos triángulos rectángulos y las fracciones al lado
def plot_two_triangles(angle_radians, hypotenuse1, hypotenuse2):
    # Cálculo de los lados para ambos triángulos
    opposite1 = np.sin(angle_radians) * hypotenuse1
    adjacent1 = np.cos(angle_radians) * hypotenuse1
    
    opposite2 = np.sin(angle_radians) * hypotenuse2
    adjacent2 = np.cos(angle_radians) * hypotenuse2
    
    # Configuración de las subgráficas
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    
    # Subgráfico 1: Dibujar los triángulos
    ax[0].plot([0, adjacent1], [0, 0], 'r', lw=3, label='CA')
    ax[0].plot([adjacent1, adjacent1], [0, opposite1], 'b', lw=3, label='CO')
    ax[0].plot([0, adjacent1], [0, opposite1], 'g', lw=3, label='H')
    
    ax[0].plot([0, adjacent2], [0, 0], 'orange', lw=3, label='CA ')
    ax[0].plot([adjacent2, adjacent2], [0, opposite2], 'purple', lw=3, label='CO')
    ax[0].plot([0, adjacent2], [0, opposite2], 'pink', lw=3, label='H')
    
    ax[0].set_xlim(-0.5, max(adjacent1, adjacent2) + 0.5)
    ax[0].set_ylim(-0.5, max(opposite1, opposite2) + 0.5)
    ax[0].set_aspect('equal')
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("Dos triángulos con diferentes hipotenusas")
    
    # Subgráfico 2: Mostrar las fracciones y cocientes de coseno y seno
    ax[1].axis('off')  # Ocultamos los ejes
    fraction_of_pi = angle_radians / np.pi  # Convertir ángulo a múltiplos de π
    
    # Cálculo del cociente de coseno (adyacente / hipotenusa)
    cos_cociente1 = adjacent1 / hypotenuse1
    cos_cociente2 = adjacent2 / hypotenuse2
    
    # Cálculo del cociente de seno (opuesto / hipotenusa)
    sin_cociente1 = opposite1 / hypotenuse1
    sin_cociente2 = opposite2 / hypotenuse2
    
    # Texto de coseno en una sola línea con colores
    ax[1].text(0.1, 0.7, f"cos({fraction_of_pi:.2f}π) =", fontsize=14, color='black')
    ax[1].text(0.4, 0.7, f"{adjacent1:.2f}", fontsize=14, color='red')
    ax[1].text(0.5, 0.7, "/", fontsize=14, color='black')
    ax[1].text(0.55, 0.7, f"{hypotenuse1:.2f}", fontsize=14, color='green')
    ax[1].text(0.65, 0.7, f"= {cos_cociente1:.2f}=", fontsize=14, color='black')
    
    ax[1].text(0.9, 0.7, f"{adjacent2:.2f}", fontsize=14, color='orange')
    ax[1].text(1., 0.7, "/", fontsize=14, color='black')
    ax[1].text(1.01, 0.7, f"{hypotenuse2:.2f}", fontsize=14, color='pink')
    ax[1].text(1.11, 0.7, f"= {cos_cociente2:.2f}", fontsize=14, color='black')
    
    # Texto de seno en una sola línea con colores
    ax[1].text(0.1, 0.6, f"sin({fraction_of_pi:.2f}π) =", fontsize=14, color='black')
    ax[1].text(0.4, 0.6, f"{opposite1:.2f}", fontsize=14, color='blue')
    ax[1].text(0.5, 0.6, "/", fontsize=14, coanalizar_diferencia_de_cuboslor='black')
    ax[1].text(0.55, 0.6, f"{hypotenuse1:.2f}", fontsize=14, color='green')
    ax[1].text(0.65, 0.6, f"= {sin_cociente1:.2f}=", fontsize=14, color='black')
    
    ax[1].text(0.9, 0.6, f"{opposite2:.2f}", fontsize=14, color='purple')
    ax[1].text(1., 0.6, "/", fontsize=14, color='black')
    ax[1].text(1.01, 0.6, f"{hypotenuse2:.2f}", fontsize=14, color='pink')
    ax[1].text(1.11, 0.6, f"= {sin_cociente2:.2f}", fontsize=14, color='black')
    
    plt.show()

# Función principal para generar la animación de "Los Pollitos"
def los_pollitos_animation(rate=22050, duration=0.5, output_file='los_pollitos_animation.mp4'):
    """
    Genera una animación de la superposición de ondas sinusoidales correspondientes
    a la melodía de 'Los Pollitos' y la guarda como archivo .mp4.

    Parámetros:
    - rate: Tasa de muestreo en Hz (por defecto 22050)
    - duration: Duración de cada nota en segundos (por defecto 0.5)
    - output_file: Nombre del archivo de salida (por defecto 'los_pollitos_animation.mp4')
    """
    # Eje de tiempo
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)

    # Frecuencias de las notas musicales (en Hz)
    notes = {
        'C': 261.63,  # Do
        'D': 293.66,  # Re
        'E': 329.63,  # Mi
        'F': 349.23,  # Fa
        'G': 392.00,  # Sol
        'A': 440.00,  # La
        'B': 493.88,  # Si
        'C_high': 523.25  # Do (una octava más alta)
    }

    # Función para generar una onda sinusoidal para una nota
    def generate_wave(note):
        frequency = notes[note]
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Amplitud de 0.5 para evitar saturación
        return wave

    # Melodía de "Los Pollitos" (simplificada)
    melody = [
        'E', 'E', 'F', 'G', 'G', 'F', 'E', 'D',  # "Los pollitos dicen pío, pío, pío"
        'C', 'C', 'D', 'E', 'E', 'D', 'D',       # "cuando tienen hambre"
        'E', 'E', 'F', 'G', 'G', 'F', 'E', 'D',  # "y cuando tienen frío"
        'C', 'C', 'D', 'E', 'D', 'C'             # "La mamá les busca el maíz y el trigo"
    ]

    # Generar las ondas para cada nota en la melodía
    waves = [generate_wave(note) for note in melody]

    # Crear la figura para la animación
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)

    # Configuración de los ejes
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Superposición de las ondas sinusoidales de 'Los Pollitos'")

    # Inicialización de la gráfica
    def init():
        line.set_data([], [])
        return line,

    # Función para actualizar la animación
    def update(frame):
        # Superponer las ondas de las primeras `frame+1` notas
        current_wave = np.sum(waves[:frame+1], axis=0) / (frame + 1)  # Normalizar la amplitud
        line.set_data(t, current_wave)
        return line,

    # Crear la animación
    ani = FuncAnimation(fig, update, frames=len(waves), init_func=init, blit=True, interval=500)

    # Guardar la animación como archivo .mp4
    ani.save(output_file, writer='ffmpeg')

    # Mostrar la animación
    plt.show()

# ---------------------------------------------------------------
# Librería Python 8
# ---------------------------------------------------------------
def plot_sucesion(expr_inicial: str = '1/n', terminos_inicial: int = 20, max_terminos: int = 100):
    """
    Interfaz interactiva en Jupyter para graficar una sucesión definida por una expresión en n.

    Parámetros:
      expr_inicial (str): Expresión de la sucesión en función de n (p.ej. '1/n' o 'np.sin(n)').
      terminos_inicial (int): Número inicial de términos a graficar.
      max_terminos (int): Límite máximo de términos en el slider.
    """
    def _actualizar(expr: str, N: int):
        n = np.arange(1, N + 1)
        try:
            y = eval(expr, {'n': n, 'np': np})
        except Exception as e:
            print(f"Error al evaluar '{expr}': {e}")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(n, y, marker='o', linestyle='-')
        plt.xlabel('n')
        plt.ylabel(expr)
        plt.title(f'Sucesión: {expr}')
        plt.grid(True)
        plt.show()

    # Widgets
    expr_widget = widgets.Text(value=expr_inicial, description='Expresión:', layout=widgets.Layout(width='60%'))
    terminos_widget = IntSlider(value=terminos_inicial, min=1, max=max_terminos, step=1,
                                 description='Términos:', continuous_update=False)

    # Salida enlazada
    out = interactive_output(_actualizar, {'expr': expr_widget, 'N': terminos_widget})

    # Mostrar interfaz
    display(VBox([HBox([expr_widget, terminos_widget]), out]))

def graficar_fibonacci(n: int = 20):
    """
    Calcula y grafica los primeros n términos de la sucesión de Fibonacci.

    Parámetros:
      n (int): Número de términos de la sucesión a generar.
    """
    if n <= 0:
        print("El número de términos debe ser mayor que 0.")
        return
    # Generar sucesión de Fibonacci
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    fib = fib[:n]

    # Índices
    indices = np.arange(1, n+1)

    # Graficar
    plt.figure(figsize=(8,4))
    plt.plot(indices, fib, marker='o', linestyle='-')
    plt.xlabel('n')
    plt.ylabel('Fibonacci(n)')
    plt.title(f'Sucesión de Fibonacci (primeros {n} términos)')
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------------------------
# Funciones Actividad 8 (Sucesiones y TICS)
# ------------------------------------------------------------------------------------
def find_min_n(T, start=2, max_n=10000):
    """
    Encuentra el valor mínimo de n tal que E(n) = n/(ln(n))^2 >= T.

    Parámetros:
    - T (float): Umbral crítico de eficiencia.
    - start (int): Valor de inicio de la búsqueda (por defecto 2).
    - max_n (int): Límite superior de búsqueda (por defecto 10_000_000).

    Retorna:
    - int o None: El valor mínimo de n que cumple la condición, o None si no se encuentra.
    """
    n = max(start, 2)
    while n <= max_n:
        if n / (np.log(n) ** 2) >= T:
            return n
        n += 1
    return None


def plot_efficiency(n_max, T):
    """
    Grafica la eficiencia E(n) = n/(ln(n))^2 para n en [2, n_max]
    y marca el umbral T y el punto crítico n mínimo.

    Parámetros:
    - n_max (int): Valor máximo de n en la gráfica.
    - T (float): Umbral crítico de eficiencia.
    """
    x = np.arange(2, n_max + 1)
    y = x / (np.log(x) ** 2)

    n_crit = find_min_n(T, start=2, max_n=n_max)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, lw=2, label=r'$E(n)=\frac{n}{(\ln n)^2}$')
    ax.axhline(T, color='red', linestyle='--', label=f'$T = {T}$')

    if n_crit is not None:
        y_crit = n_crit / (np.log(n_crit) ** 2)
        ax.axvline(n_crit, color='green', linestyle='-.', label=f'$n_{{min}} = {n_crit}$')
        ax.scatter([n_crit], [y_crit], color='green', zorder=5)
        ax.text(n_crit, y_crit * 1.05, f'$n={n_crit}$', ha='center', color='green')

    ax.set_xlim(2, n_max + int(0.05 * n_max))
    ax.set_ylim(0, max(y.max(), T * 1.2))
    ax.set_xlabel('n (nodos)')
    ax.set_ylabel(r'$E(n)=\frac{n}{(\ln n)^2}$')
    ax.set_title('Eficiencia relativa vs. número de nodos')
    ax.grid(True)
    ax.legend()
    plt.show()


def interactive_efficiency(default_n=500, default_T=100, max_n=10000):
    """
    Configura y despliega una interfaz interactiva para explorar E(n).

    Parámetros:
    - default_n (int): Valor inicial del slider para n máximo.
    - default_T (float): Valor inicial del slider para el umbral T.
    - max_n (int): Valor máximo permitido en el slider de n.
    """
    n_slider = IntSlider(value=default_n,
                         min=2,
                         max=max_n,
                         step=10,
                         description='n max:')

    T_slider = FloatSlider(value=default_T,
                           min=1,
                           max=10 * default_T,
                           step=1,
                           description='T:')

    ui = VBox([HBox([n_slider, T_slider])])
    out = interactive_output(plot_efficiency, {'n_max': n_slider, 'T': T_slider})

    display(ui, out)