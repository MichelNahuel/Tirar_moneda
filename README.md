# Tirar moneda

## Descripción

**Tirar moneda** es una herramienta interactiva de simulación estadística que
ilustra, de forma incremental y visual, cómo emergen tres distribuciones de
probabilidad clásicas a partir de un experimento tan simple como lanzar una
moneda. La aplicación, construida sobre `matplotlib`, permite ir agregando
repeticiones «a voluntad» y observar en tiempo real la convergencia de los
histogramas hacia sus densidades teóricas, junto con estadísticos descriptivos
y una prueba de bondad de ajuste de Kolmogórov–Smirnov para cada experimento.

Los tres experimentos son:

1. **Binomial → Normal (Teorema Central del Límite).**
   Se lanza una moneda `N` veces (100 por defecto) y se cuenta la cantidad de
   caras. Repetido muchas veces, el histograma de conteos se aproxima a una
   distribución normal de media `N·p` y varianza `N·p·(1−p)`.

2. **Geométrica → cola larga.**
   Se lanza una moneda hasta obtener la primera cara y se registra cuántos
   lanzamientos fueron necesarios. La distribución resultante es geométrica:
   decae rápidamente y presenta una cola larga de valores grandes. En escala
   log-log se aproxima a una recta en el rango observable.

3. **Chi-cuadrado.**
   Se toman los conteos de caras del experimento 1, se estandarizan
   (`Z = (X − N·p) / √(N·p·(1−p))`, aproximadamente `N(0, 1)` por el TCL), se
   agrupan en bloques de `k` y se suma el cuadrado de cada bloque. La suma sigue
   una distribución chi-cuadrado con `k` grados de libertad.

El proyecto tiene fines didácticos: sirve para explorar visualmente la
convergencia de distribuciones muestrales y el efecto del tamaño de muestra.

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`:
  - `numpy` — generación de números aleatorios y cálculo vectorizado
  - `matplotlib` — gráficos e interfaz interactiva (widgets)
  - `scipy` — densidades teóricas, asimetría y prueba de Kolmogórov–Smirnov

## Instalación

```bash
git clone https://github.com/MichelNahuel/Tirar_moneda.git
cd Tirar_moneda

python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

```bash
python Tirar_moneda.py
```

Se abre una ventana con tres paneles de gráficos (uno por experimento) y una
fila de controles en la parte inferior. La aplicación arranca vacía: hay que
agregar repeticiones para que aparezcan los histogramas.

### Flujo básico

1. Escribir en el campo **«Agregar:»** la cantidad de repeticiones a simular
   (por defecto 1000) y pulsar **«Actualizar»** (o Enter en el campo).
2. Repetir el paso anterior las veces que se quiera; las repeticiones se
   **acumulan** y los tres gráficos se redibujan con el total.
3. Observar cómo cada histograma se ajusta a su curva teórica (en rojo) a
   medida que crece el número de repeticiones.

### Controles disponibles

| Control | Función | Valor inicial |
|---|---|---|
| **Agregar:** | Cantidad de repeticiones nuevas a simular al pulsar «Actualizar». | 1000 |
| **Actualizar** | Ejecuta la simulación y redibuja. | — |
| **Tiradas exp.1:** | Número de lanzamientos por repetición en los experimentos 1 y 3. Cambiarlo **reinicia** los datos acumulados. | 100 |
| **Prob. cara:** | Probabilidad de cara `p` (entre 0 y 1). Cambiarla **reinicia** los datos acumulados. | 0.5 |
| **Bins hist.:** | Cantidad de barras de los histogramas (experimentos 1 y 3). | 30 |
| **Resol. log-log:** | Cantidad de puntos con los que se agrupa la cola de la geométrica (experimento 2). | 40 |
| **Grados libertad (χ²):** | Tamaño `k` de los bloques del experimento 3. | 5 |
| **Log / Lineal** (radio, arriba a la derecha) | Escala de los ejes del experimento 2. | Log |
| **Modo oscuro / claro** (botón, arriba a la izquierda) | Alterna el tema de la interfaz. | Claro |

Si un campo recibe un valor no válido (texto no numérico, vacío), la aplicación
lo avisa en la línea de estado inferior y usa el valor por defecto.

### Lectura de los paneles

- **Panel 1 — «Caras en N tiradas»**: histograma de conteos de caras y, en rojo,
  la normal teórica escalada. Debajo: media (`μ`), varianza (`σ²`), asimetría
  (`asim`) y el estadístico `D` y el `p-valor` de la prueba de Kolmogórov–Smirnov
  contra la normal.
- **Panel 2 — «Lanzamientos hasta la primera cara»**: dispersión de frecuencias
  y, en rojo, la geométrica teórica. Alternar entre escala lineal y log-log con
  el selector. Mismos estadísticos que el panel 1, contra la geométrica.
- **Panel 3 — «χ² con k=...»**: histograma de los valores chi-cuadrado y la
  densidad teórica con `k` grados de libertad, con sus estadísticos y la prueba
  KS correspondiente.

La línea de estado inferior indica el total de repeticiones simuladas.

## Estructura del proyecto

```
Tirar_moneda/
├── Tirar_moneda.py     # Simulación, cálculos estadísticos e interfaz matplotlib
├── requirements.txt
└── README.md
```

### Funciones y componentes principales

| Elemento | Rol |
|---|---|
| `experimento_binomial(n, tiradas, p)` | Lanza `tiradas` monedas, cuenta caras, repite `n` veces. |
| `experimento_geometrico(n, p)` | Cantidad de lanzamientos hasta la primera cara, repetido `n` veces. |
| `App.calcular_chi2(k)` | Estandariza los conteos de caras, los agrupa en bloques de `k` y suma cuadrados. |
| `App.binning_log(datos, n_bins)` | Agrupa la cola de la geométrica en bins logarítmicos para el gráfico log-log. |
| `App.agregar(...)` | Simula nuevas repeticiones y las concatena a los datos acumulados. |
| `App.dibujar(...)` | Recalcula estadísticos (con caché) y redibuja los tres paneles. |
