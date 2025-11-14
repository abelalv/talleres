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
    T = 8 * D + B
    return D, T

# Función para mostrar la gráfica de la temperatura y el flujo crítico
def mostrar_temperatura(B):
    D, T = calcular_temperatura(B)
    
    plt.figure(figsize=(8, 4))
    plt.plot(D, T, 'm-', lw=2, label='T(D) = 8D + B')
    plt.axhline(50, color='k', linestyle='--', label='T_c = 50°C')
    plt.xlabel('Flujo de datos D')
    plt.ylabel('Temperatura T (°C)')
    plt.title('Función de Temperatura del Cable')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 101, 10))
    plt.show()

    # Cálculo del flujo crítico D_c
    D_c = (50 - B) / 8
    print(f'Para B = {B:.2f}, el flujo crítico es D_c = {D_c:.2f}')

# Función para calcular el desgaste del cable en función del tiempo t
def calcular_desgaste():
    t_vals = np.linspace(0, 10, 400)
    d = -9/10 * t_vals + 10
    return t_vals, d

# Función para mostrar la gráfica del desgaste del cable
def mostrar_desgaste(D_c):
    t_vals, d = calcular_desgaste()
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, d, 'b-', lw=2, label='d(t) = -9/10t + 10')
    plt.axhline(D_c, color='r', linestyle='--', label=f'D_c = {D_c:.2f}')
    plt.xlabel('Tiempo t (años)')
    plt.ylabel('Capacidad D(t)')
    plt.title('Función de Capacidad máxima de transmisión del Cable')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.show()

# Función para la interacción con el slider de temperatura
def interactuar_con_temperatura():
    return interact(
        mostrar_temperatura, 
        B=FloatSlider(value=10, min=0, max=20, step=0.5, description='Parámetro B')
    )

# Función para la interacción con el slider de desgaste
def interactuar_con_desgaste():
    return interact(
        mostrar_desgaste, 
        D_c=FloatSlider(value=5, min=0, max=10, step=0.1, description='Capacidad Crítica D_c')
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
    plt.title('Flexión de la viga en voladizo')
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
    K_slider = IntSlider(value=1000, min=500, max=2000, step=100, description="K:")
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

# Funciones geométricas

def push_radially(pt, factor=1.15):
    """Desplaza un punto radialmente hacia fuera (para ubicar etiquetas)."""
    return np.array(pt) * factor


def surface_coverage_distance(h, R=6378):
    """Distancia lineal sobre la superficie visible desde altura h."""
    return np.sqrt((R + h)**2 - R**2)


def required_threshold(D, m):
    """Umbral mínimo de cobertura requerido."""
    return D / 2 + m


def view_angle(h, R=6378):
    """Ángulo de visión en radianes desde el centro terrestre."""
    return np.arccos(R / (R + h))

# Gráfico del sistema

def plot_satellite(D, h, m):
    """
    Dibuja Tierra, Quibdó–Mitú, satélite y zona de cobertura.
    Parámetros:
        D : distancia superficial entre ciudades (km)
        h : altura orbital (km)
        m : margen de cobertura (km)
    """
    R = 6378
    theta0 = 7.6 * np.pi / 180
    d = surface_coverage_distance(h, R)
    th_view = view_angle(h, R)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # --- Tierra ---
    ax.add_patch(plt.Circle((0, 0), R, color='lightblue', alpha=0.4, label='Tierra'))

    # --- Ciudades ---
    half = theta0 / 2
    Q = [R * np.cos(half), R * np.sin(half)]
    M = [R * np.cos(-half), R * np.sin(-half)]
    ax.plot(*Q, 'ko')
    ax.plot(*M, 'ko')
    ax.text(*push_radially(Q), 'Quibdó', fontsize=9)
    ax.text(*push_radially(M), 'Mitú', fontsize=9)

    # --- Satélite ---
    sat = [R + h, 0]
    ax.plot(*sat, 'ro')

    # Etiqueta ligeramente más arriba
    label_offset = 700  # km
    ax.text(sat[0], sat[1] + label_offset, f"SAT\nh={h:.0f} km", color='red', ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # --- Cono de cobertura ---
    phi = np.linspace(-th_view, th_view, 200)
    coverage_edge = R * np.exp(1j * phi)  # puntos del borde visible
    ax.plot(coverage_edge.real, coverage_edge.imag, 'r--', label='Zona visible')

    # --- Líneas del cono de visión ---
    ax.plot([R * np.cos(th_view), R + h],
            [R * np.sin(th_view), 0], 'r:', alpha=0.7)
    ax.plot([R * np.cos(-th_view), R + h],
            [R * np.sin(-th_view), 0], 'r:', alpha=0.7)

    # --- Límites del gráfico ---
    mrg = 2200
    ax.set_xlim(-R - mrg, R + mrg)
    ax.set_ylim(-R - mrg, R + mrg)
    ax.set_xlabel('km')
    ax.set_ylabel('km')

    # --- Información ---
    ax.set_title(f"Distancia visible d={d:.0f} km — Requerida ≥ {required_threshold(D, m):.0f} km",
                 fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True)
    plt.show()

# Interfaz interactiva

def create_widgets():
    """Crea los widgets de entrada."""
    R = 6378
    theta0 = 7.6 * np.pi / 180
    D_pred = R * theta0
    D_text = FloatText(value=np.round(D_pred, 2), description='D (km):')
    h_slider = FloatSlider(value=500, min=100, max=1200, step=10, description='Altura h (km):')
    m_slider = FloatSlider(value=0, min=0, max=300, step=10, description='Margen m (km):')
    return D_text, h_slider, m_slider


def build_interface_satellite():
    """Construye y muestra la interfaz completa."""
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

def interactivo_desfase_seno(v = 0):
    """
    Muestra un control interactivo (slider) que permite variar el desfase
    en la función seno y visualizar su efecto en tiempo real.
    """
    desfase_slider = FloatSlider(min=0, max=4*np.pi, step=0.1, value=v, description='Desfase')
    interact(plot_sine, desfase=desfase_slider)

def graficar_seno_coseno_desfase(desfase):
    """
    Grafica el coseno fijo y el seno desplazado por un desfase dado.
    Permite visualizar que cos(x) = sin(x + π/2).
    """
    x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
    y_cos = np.cos(x)
    y_seno = np.sin(x + desfase)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_cos, label=r"$\cos(x)$ (fija)", color='blue', linestyle='--', alpha=0.7)
    plt.plot(x, y_seno, label=fr"$\sin(x + {desfase:.2f})$", color='purple')

    plt.title(fr"Comparación entre $\sin(x + \phi)$ y $\cos(x)$   (Desfase = {desfase:.2f} rad)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-3*np.pi, 3*np.pi)
    plt.xticks(
        [-3*np.pi, -2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 3*np.pi],
        [r"$-3\pi$", r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"]
    )
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def interactivo_seno_coseno_desfase(valor_inicial=0):
    """
    Crea un control interactivo para explorar la relación entre seno y coseno
    mediante el desfase. El coseno permanece fijo mientras el seno se desplaza.
    """
    desfase_slider = FloatSlider(
        min=-2*np.pi, max=2*np.pi, step=0.1,
        value=valor_inicial, description='Desfase'
    )
    interact(graficar_seno_coseno_desfase, desfase=desfase_slider)


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

def interactivo_seno():
    """
    Crea un control interactivo que permite modificar los parámetros
    de la función seno generalizada y observar los efectos en la gráfica.

    Parámetros controlados:
    - Amplitud (A): altura máxima de la onda.
    - Frecuencia (B): número de ciclos por unidad.
    - Desfase (C): desplazamiento horizontal (en radianes).
    - Constante (D): desplazamiento vertical.

    Utiliza internamente la función graficar_seno(A, B, C, D).
    """
    interact(
        graficar_seno,
        amplitud=(1.0, 10.0, 0.1),
        frecuencia=(0.1, 5.0, 0.1),
        desfase=(-np.pi, np.pi, 0.1),
        constante=(-5.0, 5.0, 0.1)
    )

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

def interactivo_coseno():
    """
    Crea un control interactivo que permite modificar los parámetros
    de la función coseno generalizada y observar su efecto en la gráfica.

    Parámetros controlados:
    - amplitud (A): controla la altura máxima de la onda.
    - frecuencia (B): controla el número de ciclos por unidad.
    - desfase (C): desplazamiento horizontal (en radianes).
    - constante (D): desplazamiento vertical.

    Esta función utiliza internamente `graficar_coseno(A, B, C, D)` definida en la librería.
    """
    interact(
        graficar_coseno,
        amplitud=(1.0, 10.0, 0.1),
        frecuencia=(0.1, 5.0, 0.1),
        desfase=(-np.pi, np.pi, 0.1),
        constante=(-5.0, 5.0, 0.1)
    )

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

def interactivo_tangente():
    """
    Crea un control interactivo que permite modificar los parámetros
    de la función tangente generalizada y observar su efecto en la gráfica.

    Parámetros controlados:
    - amplitud (A): controla la escala vertical de la tangente.
    - frecuencia (B): controla el número de ciclos por unidad.
    - desfase (C): desplazamiento horizontal (en radianes).
    - constante (D): desplazamiento vertical.

    Esta función utiliza internamente `graficar_tangente(A, B, C, D)` definida en la librería.
    """
    interact(
        graficar_tangente,
        amplitud=(1.0, 5.0, 0.1),
        frecuencia=(0.1, 5.0, 0.1),
        desfase=(-np.pi, np.pi, 0.1),
        constante=(-5.0, 5.0, 0.1)
    )

def grados_a_radianes(grados):
    """
    Convierte grados a radianes y muestra el resultado en texto y en una gráfica sencilla.
    """
    radianes = grados * np.pi / 180
    
    # Mostrar el resultado numérico
    print(f"{grados:.2f}° equivalen a {radianes:.4f} radianes")


def interactive_grados_a_radianes(valor_inicial=0):
    """
    Crea un control interactivo que permite convertir grados a radianes.
    """
    slider = FloatSlider(
        min=0, max=360, step=1, value=valor_inicial,
        description='Grados', continuous_update=True
    )
    interact(grados_a_radianes, grados=slider)

# Definición de funciones trigonométricas recíprocas
def sec(x):
    """Función secante"""
    return np.where(np.cos(x) != 0, 1 / np.cos(x), np.nan)

def csc(x):
    """Función cosecante"""
    return np.where(np.sin(x) != 0, 1 / np.sin(x), np.nan)

def cot(x):
    """Función cotangente"""
    return np.where(np.tan(x) != 0, 1 / np.tan(x), np.nan)


def graficar_funciones_trigonometricas_extra():
    """
    Grafica las funciones trigonométricas recíprocas:
    secante, cosecante y cotangente, en el intervalo [-2π, 2π].
    """

    # Rango de valores de x
    x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)

    # Asociar cada función con su color y nombre
    funciones = {
        r"$\sec(x)$": (sec, 'royalblue'),
        r"$\csc(x)$": (csc, 'seagreen'),
        r"$\cot(x)$": (cot, 'darkviolet')
    }

    # Crear figura con 3 subgráficas verticales
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Iterar sobre funciones y ejes
    for ax, (nombre, (f, color)) in zip(axes, funciones.items()):
        y = f(x)
        ax.plot(x, y, color=color, linewidth=2, label=nombre)

        # Ejes de referencia
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)

        # Límites y formato
        ax.set_ylim(-10, 10)
        ax.set_title(f"Gráfica de {nombre}", fontsize=15)
        ax.legend()
        ax.grid(True)

        # Eje x en radianes
        ax.set_xticks([
            -2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0,
            np.pi/2, np.pi, 3*np.pi/2, 2*np.pi
        ])
        ax.set_xticklabels([
            r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$-\pi$', r'$-\frac{\pi}{2}$',
            '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'
        ])

    plt.xlabel("x (radianes)", fontsize=12)
    plt.tight_layout()
    plt.show()

def identidad_pitagorica(theta_deg: float = 0):
    """
    Visualiza la identidad pitagórica: sin²(θ) + cos²(θ) = 1
    mediante un círculo unitario y el triángulo correspondiente
    al ángulo θ.

    Parámetros:
    -----------
    theta_deg : float
        Ángulo en grados (0° a 360°)
    """
    
    # Conversión a radianes
    theta_rad = np.deg2rad(theta_deg)

    # Cálculo de seno y coseno
    seno = np.sin(theta_rad)
    coseno = np.cos(theta_rad)
    identidad = seno**2 + coseno**2  # debería ser ≈ 1

    # Configuración de la figura
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Dibujar el círculo unitario
    circle = plt.Circle((0, 0), 1, color='lightblue', fill=False, linewidth=2)
    ax.add_patch(circle)

    # Dibujar triángulo (radio, proyección en x, proyección en y)
    plt.plot([0, coseno], [0, seno], color='k', lw=2, label='Radio = 1')
    plt.plot([coseno, coseno], [0, seno], 'g--', lw=1)
    plt.plot([0, coseno], [0, 0], 'b--', lw=1)

    # Dibujar punto sobre el círculo
    plt.scatter(coseno, seno, color='red', s=60, zorder=5)

    # Etiquetas explicativas
    plt.text(0.05, 1.1, f"$\\sin^2(\\theta) + \\cos^2(\\theta) = {identidad:.2f}$",
             fontsize=12, color='darkred')
    plt.text(coseno / 2, -0.15, f"$\\cos(\\theta) = {coseno:.2f}$", fontsize=11, color='green')
    plt.text(0.05, seno / 2, f"$\\sin(\\theta) = {seno:.2f}$", fontsize=11, color='blue')

    # Decoración de los ejes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"Identidad Pitagórica — $\\theta = {theta_deg}^\\circ$", fontsize=14)
    plt.legend(loc='upper right')
    plt.show()


def interactivo_identidad_pitagorica():
    """
    Crea un control interactivo para explorar la identidad pitagórica
    variando el ángulo θ en grados.
    """
    interact(identidad_pitagorica, theta_deg=(0, 360, 1))

def graficar_inversas():
    """
    Grafica las funciones trigonométricas inversas: arcsin(x), arccos(x) y arctan(x)
    con escalas iguales y etiquetas matemáticas en los ejes.
    """

    # --- RANGOS DE LAS FUNCIONES ---
    x_sin_cos = np.linspace(-1, 1, 1000)     # Dominio válido para arcsin y arccos
    x_tan = np.linspace(-10, 10, 1000)       # Rango amplio para arctan

    # --- CÁLCULO DE LAS FUNCIONES ---
    y_arcsin = np.arcsin(x_sin_cos)
    y_arccos = np.arccos(x_sin_cos)
    y_arctan = np.arctan(x_tan)

    # --- CREACIÓN DE LA FIGURA ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    funciones = [
        (x_sin_cos, y_arcsin,  r"$\arcsin(x)$", "Función Inversa del Seno",   [-np.pi/2, 0, np.pi/2], [r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"], 'blue'),
        (x_sin_cos, y_arccos,  r"$\arccos(x)$", "Función Inversa del Coseno", [0, np.pi/2, np.pi], ["0", r"$\frac{\pi}{2}$", r"$\pi$"], 'green'),
        (x_tan,     y_arctan,  r"$\arctan(x)$", "Función Inversa de la Tangente", [-np.pi/2, 0, np.pi/2], [r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"], 'purple')
    ]

    # --- GRAFICAR CADA FUNCIÓN ---
    for ax, (x, y, label, titulo, yticks, ylabels, color) in zip(axes, funciones):
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.set_title(titulo, fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.legend(fontsize=12, loc='best')

    # --- AJUSTAR Y MOSTRAR ---
    plt.tight_layout()
    plt.show()

# Función interactiva para graficar y resolver la ecuación seno: sin(θ) = A
def resolver_seno(A, tolerancia=0.2):
    """
    Grafica la ecuación sin(θ) = A y muestra las soluciones aproximadas
    en el rango [-2π, 2π]. Agrupa soluciones cercanas dentro de una tolerancia.
    """
    # Definir el rango de θ en el intervalo [-2π, 2π]
    x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)
    y = np.sin(x)

    # Verificar que A esté dentro del rango válido [-1, 1]
    if A < -1 or A > 1:
        print("No existen soluciones reales para |A| > 1, ya que el seno está definido entre -1 y 1.")
        return

    # Crear la figura
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=r"$\sin(\theta)$", color='blue', linewidth=2)
    plt.axhline(y=A, color='red', linestyle='--', label=f"$y = {A}$")

    # Encontrar los puntos donde sin(θ) ≈ A
    sol_indices = np.where(np.isclose(y, A, atol=0.01))[0]
    sol_x = x[sol_indices]

    # --- Agrupar soluciones cercanas ---
    sol_x_agrupadas = []
    if len(sol_x) > 0:
        grupo_actual = [sol_x[0]]
        for s in sol_x[1:]:
            if abs(s - grupo_actual[-1]) < tolerancia:
                grupo_actual.append(s)
            else:
                sol_x_agrupadas.append(np.mean(grupo_actual))
                grupo_actual = [s]
        sol_x_agrupadas.append(np.mean(grupo_actual))
    sol_x = np.array(sol_x_agrupadas)

    # Mostrar las soluciones gráficamente
    for sol in sol_x:
        plt.plot(sol, A, 'ro')
        plt.text(sol, A + 0.1, f"$\\theta = {sol:.2f}$", fontsize=11, color='red', ha='center')

    # Ajustes del gráfico
    plt.title(fr"Soluciones de $\sin(\theta) = {A}$ en el intervalo $[-2\pi, 2\pi]$", fontsize=14)
    plt.xlabel(r"$\theta$", fontsize=12)
    plt.ylabel(r"$\sin(\theta)$", fontsize=12)
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-1.5, 1.5)
    plt.xticks(
        [-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        [r"$-2\pi$", r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$", "0",
         r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

    # Imprimir soluciones numéricas
    if len(sol_x) > 0:
        print(f"Soluciones aproximadas para sin(θ) = {A}:")
        for sol in sol_x:
            print(f"  θ ≈ {sol:.2f} rad")
    else:
        print("No se encontraron soluciones dentro del rango mostrado.")


def interactivo_resolver_seno():
    """
    Crea un control interactivo para explorar la ecuación sin(θ) = A.
    """
    interact(resolver_seno, A=(-1.0, 1.0, 0.1))

# Función mejorada para graficar una ecuación trigonométrica cuadrática en seno
def graficar_funcion_trigonometrica(rango_x=(0, 2 * np.pi), num_puntos=1000, figsize=(9, 6)):
    """
    Grafica la ecuación 2·sin²(x) − sin(x) − 1 sobre el rango dado,
    mostrando las soluciones en el intervalo [0, 2π].

    Parámetros:
      - rango_x (tuple): Tupla (x_min, x_max) para el dominio.
      - num_puntos (int): Cantidad de puntos para el muestreo en x.
      - figsize (tuple): Tamaño de la figura (ancho, alto).
    """
    # Definir el rango de x y calcular la función
    x = np.linspace(rango_x[0], rango_x[1], num_puntos)
    y = 2 * np.sin(x)**2 - np.sin(x) - 1

    # Crear la figura
    plt.figure(figsize=figsize)
    plt.plot(x, y, color='royalblue', linewidth=2.5, label=r"$f(x) = 2\sin^2(x) - \sin(x) - 1$")
    
    # Ejes principales
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='gray', linewidth=0.8)

    # Soluciones teóricas (en [0, 2π])
    soluciones = [np.pi/2, 7*np.pi/6, 11*np.pi/6]
    etiquetas = [r"$\frac{\pi}{2}$", r"$\frac{7\pi}{6}$", r"$\frac{11\pi}{6}$"]

    # Marcar y anotar las soluciones
    for x_sol, label in zip(soluciones, etiquetas):
        plt.axvline(x_sol, color='crimson', linestyle='--', alpha=0.7)
        plt.plot(x_sol, 0, 'ro', markersize=8)
        plt.text(x_sol, 0.25, label, fontsize=13, color='crimson', ha='center')

    # Título y etiquetas
    plt.title(r"Resolución gráfica de $2\sin^2(x) - \sin(x) - 1 = 0$", fontsize=16, pad=15)
    plt.xlabel(r"$x$", fontsize=13)
    plt.ylabel(r"$f(x)$", fontsize=13)

    # Etiquetas personalizadas en el eje x
    plt.xticks(
        [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    )

    # Mejorar la cuadrícula
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.ylim(-2, 2)
    plt.xlim(rango_x)

    # Mostrar la gráfica
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





