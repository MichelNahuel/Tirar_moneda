"""
Simulación de dos experimentos con monedas, con una interfaz gráfica
simple (hecha con matplotlib) para ir agregando repeticiones a voluntad.

1) Binomial -> Normal:
   Se lanza una moneda 100 veces y se cuenta cuántas caras salieron.
   Se repite n veces. La distribución de esos conteos se aproxima
   a una normal (Teorema Central del Límite).

2) Geométrica -> "ley de potencia":
   Se lanza una moneda hasta que sale cara, anotando cuántos
   lanzamientos hicieron falta. Se repite n veces. La distribución
   resultante es geométrica: decae rápido al principio y tiene una
   cola larga de valores grandes. En escala log-log se ve
   aproximadamente lineal en el rango observable, En escala lineal
   decae de forma exponencial como una ley de potencia. 
   
3) Chi Cuadrado -> Distribución Chi-Cuadrado:
   Se lanza una moneda 100 veces y se cuenta cuántas caras salieron.
   Se repite n veces. Agrupar las repeticiones en bloques de k, 
   estandarizar cada conteo de caras (Z = (X-50)/5, aprox. N(0,1) por TCL)
   y suma los cuadrados de cada bloque -> Chi-Cuadrado con k Gra.Lib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.widgets import TextBox, Button, RadioButtons

RNG = np.random.default_rng()


def experimento_binomial(n: int, tiradas: int = 100, p: float = 0.5) -> np.ndarray:
    """Lanza `tiradas` monedas, cuenta caras (prob. p de cara), repite n veces."""
    return (RNG.random(size=(n, tiradas)) < p).sum(axis=1)


def experimento_geometrico(n: int, p: float = 0.5) -> np.ndarray:
    #Lanza una moneda hasta obtener cara, repite n veces.
    #Devuelve el número de lanzamientos necesarios en cada repetición.
    return RNG.geometric(p=p, size=n)


class App:
    def __init__(self):
        self.caras_acum = np.array([], dtype=int)
        self.intentos_acum = np.array([], dtype=int)
        #Caracteristica de las tiradas
        self.tiradas_exp1 = 100
        self.prob_cara = 0.5

        self._cache_key = None
        self._cache_pesado = None

        self.fig = plt.figure(figsize=(17, 6))
        self.fig.canvas.supports_blit = False
        self.fig.canvas.manager.set_window_title("Evolución de distribuciones")

        # Ejes de los gráficos 
        self.ax1 = self.fig.add_axes([0.05, 0.25, 0.27, 0.65])
        self.ax2 = self.fig.add_axes([0.37, 0.25, 0.27, 0.65])
        self.ax3 = self.fig.add_axes([0.70, 0.25, 0.27, 0.65])

        # Controles: caja de texto + botón + texto de estado
        ax_box = self.fig.add_axes([0.22, 0.08, 0.10, 0.06])
        ax_bins = self.fig.add_axes([0.40, 0.08, 0.08, 0.06])
        ax_res = self.fig.add_axes([0.58, 0.08, 0.08, 0.06])
        ax_btn = self.fig.add_axes([0.72, 0.08, 0.15, 0.06])
        ax_radio = self.fig.add_axes([0.86, 0.90, 0.10, 0.08])
        ax_tema = self.fig.add_axes([0.01, 0.90, 0.09, 0.05])
        ax_gl = self.fig.add_axes([0.40, 0.16, 0.10, 0.05])
        ax_tiradas = self.fig.add_axes([0.06, 0.16, 0.10, 0.05])
        ax_prob = self.fig.add_axes([0.20, 0.16, 0.10, 0.05])


        self.tb_tiradas = TextBox(ax_tiradas, "Tiradas exp.1: ", initial="100")
        self.tb_prob = TextBox(ax_prob, "Prob. cara: ", initial="0.5")
        self.tb_tiradas.on_submit(self._cambiar_parametros_moneda)
        self.tb_prob.on_submit(self._cambiar_parametros_moneda)
        self.textbox = TextBox(ax_box, "Agregar: ", initial="1000")
        self.tb_bins = TextBox(ax_bins, "Bins hist.: ", initial="30")
        self.tb_res = TextBox(ax_res, "Resol. log-log: ", initial="40")
        self.tb_gl = TextBox(ax_gl, "Grados libertad (χ²): ", initial="5")
        self.tb_gl.on_submit(lambda _: self.dibujar())
        self.boton = Button(ax_btn, "Actualizar")
        self.radio_escala = RadioButtons(ax_radio, ("Log", "Lineal"), active=0)
        self.boton.on_clicked(self.agregar)
        self.textbox.on_submit(self.agregar)
        self.tb_bins.on_submit(lambda _: self.dibujar())
        self.tb_res.on_submit(lambda _: self.dibujar())
        self.radio_escala.on_clicked(self.dibujar)
        self.radio_escala.on_clicked(lambda _: self.dibujar())
        self.boton.on_clicked(self.agregar)
        self.textbox.on_submit(self.agregar)
        self.boton_tema = Button(ax_tema, "Modo Oscuro")
        self.boton_tema.on_clicked(self.alternar_tema)


        self.modo_oscuro = False

        self.status = self.fig.text(0.07, 0.02, "Total simulado: 0", fontsize=10)
        self.stats1 = self.fig.text(0.06, 0.235, "", fontsize=7, va="top")
        self.stats2 = self.fig.text(0.38, 0.235, "", fontsize=7, va="top")
        self.stats3 = self.fig.text(0.71, 0.235, "", fontsize=7, va="top")

        self.dibujar()
        plt.show()

    def calcular_chi2(self, k):
        n = len(self.caras_acum)
        m = n // k
        if m == 0:
            return np.array([])
        grupos = self.caras_acum[:m * k].reshape(m, k)
        mu = self.tiradas_exp1 * self.prob_cara
        sigma = np.sqrt(self.tiradas_exp1 * self.prob_cara * (1 - self.prob_cara))
        z = (grupos - mu) / sigma
        return (z ** 2).sum(axis=1)

    def _leer_entero(self, textbox, default, minimo=1):
        #Lee un entero de un TextBox con manejo de errores centralizado.
        #Si está vacío o no es un entero válido, devuelve `default` y avisa.
        texto = textbox.text.strip()
        if texto == "":
            return default
        try:
            valor = int(texto)
        except ValueError:
            self.status.set_text(f"'{texto}' no es un número entero válido, se usó {default}.")
            return default
        return max(minimo, valor)

    def _leer_float(self, textbox, default, minimo=None, maximo=None):
        texto = textbox.text.strip()
        if texto == "":
            return default
        try:
            valor = float(texto)
        except ValueError:
            self.status.set_text(f"'{texto}' no es un número válido, se usó {default}.")
            return default
        if minimo is not None:
            valor = max(minimo, valor)
        if maximo is not None:
            valor = min(maximo, valor)
        return valor

    def _cambiar_parametros_moneda(self, _event=None):
        self.tiradas_exp1 = self._leer_entero(self.tb_tiradas, 100)
        self.prob_cara = self._leer_float(self.tb_prob, 0.5, minimo=0.0, maximo=1.0)
        self.caras_acum = np.array([], dtype=int)
        self.intentos_acum = np.array([], dtype=int)
        self._cache_key = None
        self.dibujar()

    def color_grid(self):
        return "#555555" if self.modo_oscuro else "#999999"

    def alternar_tema(self, _event=None):
        self.modo_oscuro= not self.modo_oscuro
        self.dibujar()

    def aplicar_tema(self):
        if self.modo_oscuro:
            fondo, panel, texto, grid_color = "#1e1e1e", "#2b2b2b", "white", "#555555"
        else:
            fondo, panel, texto, grid_color = "white", "white", "black", "#999999"

        self.fig.patch.set_facecolor(fondo)
        self.boton_tema.label.set_text("Modo claro" if self.modo_oscuro else "Modo oscuro")

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_facecolor(panel)
            ax.tick_params(colors=texto)
            ax.xaxis.label.set_color(texto)
            ax.yaxis.label.set_color(texto)
            ax.title.set_color(texto)
            for spine in ax.spines.values():
                spine.set_color(texto)

        for widget_ax in (self.textbox.ax, self.tb_bins.ax, self.tb_res.ax, self.tb_gl.ax, self.tb_tiradas.ax, self.tb_prob.ax, self.boton.ax, self.boton_tema.ax, self.radio_escala.ax):
            widget_ax.set_facecolor(panel)
            

        self.status.set_color(texto)
        for t in (self.stats1, self.stats2, self.stats3):
            t.set_color(texto)
    
    def agregar(self, _event=None):
        texto = self.textbox.text.strip()
        try:
            n_extra = int(texto)
        except ValueError:
            self.status.set_text(f"'{texto}' no es un número entero válido.")
            self.fig.canvas.draw_idle()
            return
        if n_extra <= 0:
            self.status.set_text("Ingresá un número positivo.")
            self.fig.canvas.draw_idle()
            return

        self.caras_acum = np.concatenate([self.caras_acum, experimento_binomial(n_extra, self.tiradas_exp1, self.prob_cara)])
        self.intentos_acum = np.concatenate([self.intentos_acum, experimento_geometrico(n_extra, self.prob_cara)])
        self.dibujar()

    @staticmethod
    def binning_log(datos, n_bins):
        datos = datos[datos > 0]
        valores, conteos = np.unique(datos, return_counts=True)

        if n_bins >= len(valores):
            # No hace falta agrupar: se muestran todos los valores reales.
            return valores, conteos

        bordes = np.logspace(np.log10(valores.min()), np.log10(valores.max() + 1), n_bins + 1)
        conteos_bin, bordes = np.histogram(datos, bins=bordes)
        centros = np.sqrt(bordes[:-1] * bordes[1:])
        mask = conteos_bin > 0
        return centros[mask], conteos_bin[mask]

    def dibujar(self):
        n = len(self.caras_acum)

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.stats1.set_text("")
        self.stats2.set_text("")
        self.stats3.set_text("")

        if n > 0:
            bins_txt = self._leer_entero(self.tb_bins, 30)
            k_gl = self._leer_entero(self.tb_gl, 5)
            res_txt = self._leer_entero(self.tb_res, 40)

            mu1 = self.tiradas_exp1 * self.prob_cara
            sigma1 = np.sqrt(self.tiradas_exp1 * self.prob_cara * (1 - self.prob_cara))

            cache_key = (n, len(self.intentos_acum), bins_txt, k_gl, res_txt,
                         self.tiradas_exp1, self.prob_cara)
            if cache_key != self._cache_key:
                chi2_vals = self.calcular_chi2(k_gl)
                n_unicos = len(np.unique(self.intentos_acum))
                n_res = min(res_txt, n_unicos)
                valores, conteos = self.binning_log(self.intentos_acum, n_res) if n_unicos else (np.array([]), np.array([]))

                stats_c = (self.caras_acum.mean(), self.caras_acum.var(), stats.skew(self.caras_acum),
                           stats.kstest(self.caras_acum, "norm", args=(mu1, sigma1)) if sigma1 > 0 else None)

                stats_i = (self.intentos_acum.mean(), self.intentos_acum.var(), stats.skew(self.intentos_acum),
                           stats.kstest(self.intentos_acum, "geom", args=(self.prob_cara,))
                           if 0 < self.prob_cara < 1 else None)

                if len(chi2_vals) > 0:
                    stats_x = (chi2_vals.mean(), chi2_vals.var(), stats.skew(chi2_vals),
                               stats.kstest(chi2_vals, "chi2", args=(k_gl,)))
                else:
                    stats_x = None

                self._cache_pesado = (chi2_vals, valores, conteos, stats_c, stats_i, stats_x)
                self._cache_key = cache_key

            chi2_vals, valores, conteos, stats_c, stats_i, stats_x = self._cache_pesado

            # --- Experimento 1: Binomial -> Normal ---
            rango_caras = self.caras_acum.max() - self.caras_acum.min() + 1
            bins = min(bins_txt, rango_caras)
            self.ax1.hist(self.caras_acum,
                           bins=np.linspace(self.caras_acum.min() - 0.5, self.caras_acum.max() + 0.5, bins + 1),
                           color="steelblue", edgecolor="black")
            if sigma1 > 0:
                x1 = np.linspace(self.caras_acum.min(), self.caras_acum.max(), 200)
                ancho_bin1 = (self.caras_acum.max() - self.caras_acum.min() + 1) / bins
                self.ax1.plot(x1, n * ancho_bin1 * stats.norm.pdf(x1, mu1, sigma1),
                               color="crimson", linewidth=1.5, label="Normal teórica")
                self.ax1.legend(fontsize=7)

            mu_c, var_c, asim_c, ks_c = stats_c
            texto1 = f"μ={mu_c:.2f}  σ²={var_c:.2f}  asim={asim_c:.2f}"
            if ks_c is not None:
                texto1 += f"\nKS: D={ks_c.statistic:.3f}  p={ks_c.pvalue:.3f}"
            self.stats1.set_text(texto1)

            # --- Experimento 2: Geométrica -> ley de potencia ---
            if len(valores) > 0:
                self.ax2.scatter(valores, conteos, color="darkorange", s=15)
                if 0 < self.prob_cara < 1:
                    k_vals = np.arange(1, int(self.intentos_acum.max()) + 1)
                    self.ax2.plot(k_vals, n * stats.geom.pmf(k_vals, self.prob_cara),
                                  color="crimson", linewidth=1.2, label="Geométrica teórica")
                    self.ax2.legend(fontsize=7)

            mu_i, var_i, asim_i, ks_i = stats_i
            texto2 = f"μ={mu_i:.2f}  σ²={var_i:.2f}  asim={asim_i:.2f}"
            if ks_i is not None:
                texto2 += f"\nKS: D={ks_i.statistic:.3f}  p={ks_i.pvalue:.3f}"
            self.stats2.set_text(texto2)

            if self.radio_escala.value_selected == "Log":
                self.ax2.set_xscale("log")
                self.ax2.set_yscale("log")
                self.ax2.set_title("Lanzamientos hasta la primera cara")
                self.ax2.set_xlabel("Cantidad de lanzamientos (escala log)")
                self.ax2.set_ylabel("Frecuencia (escala log)")
            else:
                self.ax2.set_xscale("linear")
                self.ax2.set_yscale("linear")
                self.ax2.set_title("Lanzamientos hasta la primera cara")
                self.ax2.set_xlabel("Cantidad de lanzamientos (escala lineal)")
                self.ax2.set_ylabel("Frecuencia (escala lineal)")

            # --- Experimento 3: Chi-Cuadrado ---
            if len(chi2_vals) > 0:
                bins_chi2 = min(bins_txt, len(np.unique(chi2_vals)))
                self.ax3.hist(chi2_vals, bins=bins_chi2, color="mediumseagreen", edgecolor="black")
                x3 = np.linspace(max(chi2_vals.min(), 1e-6), chi2_vals.max(), 200)
                ancho_bin3 = (chi2_vals.max() - chi2_vals.min()) / bins_chi2
                self.ax3.plot(x3, len(chi2_vals) * ancho_bin3 * stats.chi2.pdf(x3, df=k_gl),
                               color="crimson", linewidth=1.5, label="χ² teórica")
                self.ax3.legend(fontsize=7)

                mu_x, var_x, asim_x, ks_x = stats_x
                self.stats3.set_text(
                    f"μ={mu_x:.2f}  σ²={var_x:.2f}  asim={asim_x:.2f}\n"
                    f"KS: D={ks_x.statistic:.3f}  p={ks_x.pvalue:.3f}"
                )

        self.ax1.set_title(f"Caras en {self.tiradas_exp1} tiradas")
        self.ax1.set_xlabel("Cantidad de caras")
        self.ax1.set_ylabel("Frecuencia")

        self.ax3.set_title(f"χ² con k={self.tb_gl.text or '5'} (agrupando exp. 1)")
        self.ax3.set_xlabel("Valor χ²")
        self.ax3.set_ylabel("Frecuencia")

        self.ax1.grid(True, linestyle="--", alpha=0.4, color=self.color_grid())
        self.ax2.grid(True, which="major", linestyle="--", alpha=0.5, color=self.color_grid())
        self.ax2.grid(True, which="minor", linestyle=":", alpha=0.2, color=self.color_grid())
        self.ax3.grid(True, linestyle="--", alpha=0.4, color=self.color_grid())

        self.status.set_text(f"Total simulado: {n} repeticiones")

        self.aplicar_tema()
        self.fig.canvas.draw()

if __name__ == "__main__":
    App()