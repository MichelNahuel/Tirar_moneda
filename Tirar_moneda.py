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

from typing import ClassVar

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
    # Paleta validada (colorblind-safe): superficies/tinta + 3 colores
    # categóricos (uno por experimento) + un acento rojo para las curvas teóricas.
    PALETTE: ClassVar[dict] = {
        "light": {
            "page": "#f9f9f7", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
            "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
            "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948",
        },
        "dark": {
            "page": "#0d0d0d", "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
            "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
            "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "red": "#e66767",
        },
    }

    def __init__(self):
        self.caras_acum = np.array([], dtype=int)
        self.intentos_acum = np.array([], dtype=int)
        #Caracteristica de las tiradas
        self.tiradas_exp1 = 100
        self.prob_cara = 0.5

        self._cache_key = None
        self._cache_pesado = None
        self.modo_oscuro = False

        plt.rcParams["font.family"] = "sans-serif"

        self.fig = plt.figure(figsize=(16.5, 8.6))
        self.fig.canvas.supports_blit = False
        self.fig.canvas.manager.set_window_title("Evolución de distribuciones")

        self.suptitle = self.fig.suptitle(
            "De la moneda a las distribuciones", fontsize=14, fontweight="bold", y=0.99
        )

        # Ejes de los gráficos
        self.ax1 = self.fig.add_axes([0.045, 0.44, 0.275, 0.41])
        self.ax2 = self.fig.add_axes([0.375, 0.44, 0.275, 0.41])
        self.ax3 = self.fig.add_axes([0.705, 0.44, 0.275, 0.41])

        # Barra superior: tema y escala del gráfico 2
        ax_tema = self.fig.add_axes([0.02, 0.91, 0.115, 0.042])
        ax_radio = self.fig.add_axes([0.865, 0.895, 0.10, 0.075])

        # Fila 1 de controles: parámetros de los experimentos
        ax_tiradas = self.fig.add_axes([0.095, 0.145, 0.135, 0.05])
        ax_prob = self.fig.add_axes([0.245, 0.145, 0.115, 0.05])
        ax_gl = self.fig.add_axes([0.375, 0.145, 0.15, 0.05])
        ax_bins = self.fig.add_axes([0.545, 0.145, 0.11, 0.05])
        ax_res = self.fig.add_axes([0.675, 0.145, 0.13, 0.05])

        # Fila 2 de controles: acción principal
        ax_box = self.fig.add_axes([0.05, 0.035, 0.16, 0.06])
        ax_btn = self.fig.add_axes([0.23, 0.035, 0.16, 0.06])

        self.tb_tiradas = TextBox(ax_tiradas, "Tiradas exp. 1", initial="100")
        self.tb_prob = TextBox(ax_prob, "Prob. cara", initial="0.5")
        self.tb_tiradas.on_submit(self._cambiar_parametros_moneda)
        self.tb_prob.on_submit(self._cambiar_parametros_moneda)

        self.tb_gl = TextBox(ax_gl, "Grados libertad (χ²)", initial="5")
        self.tb_gl.on_submit(lambda _: self.dibujar())
        self.tb_bins = TextBox(ax_bins, "Bins hist.", initial="30")
        self.tb_bins.on_submit(lambda _: self.dibujar())
        self.tb_res = TextBox(ax_res, "Resol. log-log", initial="40")
        self.tb_res.on_submit(lambda _: self.dibujar())

        self.textbox = TextBox(ax_box, "Agregar", initial="1000")
        self.textbox.on_submit(self.agregar)
        self.boton = Button(ax_btn, "Actualizar")
        self.boton.on_clicked(self.agregar)

        self.radio_escala = RadioButtons(ax_radio, ("Log", "Lineal"), active=0)
        self.radio_escala.on_clicked(lambda _: self.dibujar())

        self.boton_tema = Button(ax_tema, "Modo oscuro")
        self.boton_tema.on_clicked(self.alternar_tema)

        # Las etiquetas de los TextBox van centradas arriba del cuadro,
        # en vez de a la izquierda (default de matplotlib).
        for tb in (self.tb_tiradas, self.tb_prob, self.tb_gl, self.tb_bins,
                   self.tb_res, self.textbox):
            tb.label.set_position((0.5, 1.45))
            tb.label.set_horizontalalignment("center")
            tb.label.set_verticalalignment("bottom")

        self.status = self.fig.text(0.05, 0.005, "Total simulado: 0 repeticiones", fontsize=10)
        stats_bbox = {"boxstyle": "round,pad=0.4", "linewidth": 0.7}
        self.stats1 = self.fig.text(0.055, 0.35, "", fontsize=8, va="top",
                                     family="monospace", bbox=stats_bbox)
        self.stats2 = self.fig.text(0.385, 0.35, "", fontsize=8, va="top",
                                     family="monospace", bbox=stats_bbox)
        self.stats3 = self.fig.text(0.715, 0.35, "", fontsize=8, va="top",
                                     family="monospace", bbox=stats_bbox)

        self.dibujar()
        plt.show()

    def _estilo_borde(self, n_bins):
        #Borde entre barras del histograma. Con muchos bins cada barra ocupa
        #pocos píxeles: un borde de grosor fijo puede tapar la barra entera
        #(se "desaparece") o, si es muy fino pero sigue presente, generar
        #artefactos de moiré entre bordes vecinos. Por eso se lo achica con
        #la cantidad de bins y, pasado un umbral, se saca del todo.
        ancho = max(0.0, min(1.0, 45.0 / max(n_bins, 1)))
        if ancho < 0.15:
            return "none", 0.0
        return self.pal["surface"], ancho

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

    @property
    def pal(self):
        return self.PALETTE["dark" if self.modo_oscuro else "light"]

    def color_grid(self):
        return self.pal["grid"]

    def alternar_tema(self, _event=None):
        self.modo_oscuro = not self.modo_oscuro
        self.dibujar()

    def aplicar_tema(self):
        p = self.pal

        self.fig.patch.set_facecolor(p["page"])
        self.suptitle.set_color(p["ink"])
        self.boton_tema.label.set_text("Modo claro" if self.modo_oscuro else "Modo oscuro")

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_facecolor(p["surface"])
            ax.tick_params(colors=p["ink2"], labelsize=9)
            ax.xaxis.label.set_color(p["ink2"])
            ax.yaxis.label.set_color(p["ink2"])
            ax.xaxis.label.set_fontsize(10)
            ax.yaxis.label.set_fontsize(10)
            ax.title.set_color(p["ink"])
            ax.title.set_fontsize(12)
            ax.title.set_fontweight("bold")
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            for lado in ("bottom", "left"):
                ax.spines[lado].set_color(p["axis"])
                ax.spines[lado].set_linewidth(0.8)

        textboxes = (self.textbox, self.tb_bins, self.tb_res, self.tb_gl,
                     self.tb_tiradas, self.tb_prob)
        for tb in textboxes:
            tb.ax.set_facecolor(p["surface"])
            for spine in tb.ax.spines.values():
                spine.set_color(p["axis"])
                spine.set_linewidth(0.8)
            tb.label.set_color(p["ink2"])
            tb.label.set_fontsize(9)
            tb.text_disp.set_color(p["ink"])

        for spine in self.boton_tema.ax.spines.values():
            spine.set_color(p["axis"])
        self.boton_tema.ax.set_facecolor(p["surface"])
        self.boton_tema.color = p["surface"]
        self.boton_tema.hovercolor = p["grid"]
        self.boton_tema.label.set_color(p["ink2"])
        self.boton_tema.label.set_fontsize(9)

        for spine in self.boton.ax.spines.values():
            spine.set_color(p["blue"])
        self.boton.ax.set_facecolor(p["blue"])
        self.boton.color = p["blue"]
        self.boton.hovercolor = to_rgba(p["blue"], alpha=0.8)
        self.boton.label.set_color("#ffffff")
        self.boton.label.set_fontsize(10)
        self.boton.label.set_fontweight("bold")

        self.radio_escala.ax.set_facecolor(p["surface"])
        for spine in self.radio_escala.ax.spines.values():
            spine.set_color(p["axis"])
        for texto in self.radio_escala.labels:
            texto.set_color(p["ink2"])
            texto.set_fontsize(9)

        self.status.set_color(p["ink"])
        for t in (self.stats1, self.stats2, self.stats3):
            t.set_color(p["ink2"])
            parche = t.get_bbox_patch()
            if parche is not None:
                parche.set_facecolor(p["surface"])
                parche.set_edgecolor(p["grid"])

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

    @staticmethod
    def _limpiar_ejes(ax):
        #Saca las barras/líneas/leyenda del gráfico anterior sin usar
        #ax.clear(): clear() reconstruye también los ticks y el eje entero,
        #que es la parte más lenta de redibujar (se nota como demora al
        #apretar los botones). Al no tocar esos objetos, matplotlib solo
        #tiene que recalcular sus valores, no recrearlos.
        for patch in list(ax.patches):
            patch.remove()
        for line in list(ax.lines):
            line.remove()
        for coleccion in list(ax.collections):
            coleccion.remove()
        leyenda = ax.get_legend()
        if leyenda is not None:
            leyenda.remove()
        # Sin esto, los límites de los ejes (dataLim) quedan "pegados" al
        # rango de los datos ya removidos y sólo crecen entre redibujados.
        ax.relim()
        ax.autoscale_view()

    def dibujar(self):
        n = len(self.caras_acum)
        p = self.pal

        self._limpiar_ejes(self.ax1)
        self._limpiar_ejes(self.ax2)
        self._limpiar_ejes(self.ax3)
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
                           stats.kstest(self.caras_acum, stats.norm.cdf, args=(mu1, sigma1)) if sigma1 > 0 else None)

                stats_i = (self.intentos_acum.mean(), self.intentos_acum.var(), stats.skew(self.intentos_acum),
                           stats.kstest(self.intentos_acum, stats.geom.cdf, args=(self.prob_cara,))
                           if 0 < self.prob_cara < 1 else None)

                if len(chi2_vals) > 0:
                    stats_x = (chi2_vals.mean(), chi2_vals.var(), stats.skew(chi2_vals),
                               stats.kstest(chi2_vals, stats.chi2.cdf, args=(k_gl,)))
                else:
                    stats_x = None

                self._cache_pesado = (chi2_vals, valores, conteos, stats_c, stats_i, stats_x)
                self._cache_key = cache_key

            chi2_vals, valores, conteos, stats_c, stats_i, stats_x = self._cache_pesado

            # --- Experimento 1: Binomial -> Normal ---
            rango_caras = self.caras_acum.max() - self.caras_acum.min() + 1
            bins = min(bins_txt, rango_caras)
            borde1, ancho1 = self._estilo_borde(bins)
            self.ax1.hist(self.caras_acum,
                           bins=np.linspace(self.caras_acum.min() - 0.5, self.caras_acum.max() + 0.5, bins + 1),
                           color=p["blue"], edgecolor=borde1, linewidth=ancho1)
            if sigma1 > 0:
                x1 = np.linspace(self.caras_acum.min(), self.caras_acum.max(), 200)
                ancho_bin1 = (self.caras_acum.max() - self.caras_acum.min() + 1) / bins
                self.ax1.plot(x1, n * ancho_bin1 * stats.norm.pdf(x1, mu1, sigma1),
                               color=p["red"], linewidth=2, label="Normal teórica")
                self.ax1.legend(fontsize=8, frameon=False, labelcolor=p["ink2"])

            mu_c, var_c, asim_c, ks_c = stats_c
            texto1 = f"μ={mu_c:.2f}  σ²={var_c:.2f}  asim={asim_c:.2f}"
            if ks_c is not None:
                texto1 += f"\nKS: D={ks_c.statistic:.3f}  p={ks_c.pvalue:.3f}"
            self.stats1.set_text(texto1)

            # --- Experimento 2: Geométrica -> ley de potencia ---
            if len(valores) > 0:
                self.ax2.scatter(valores, conteos, color=p["orange"], edgecolor=p["surface"],
                                  linewidth=0.5, s=22)
                if 0 < self.prob_cara < 1:
                    k_vals = np.arange(1, int(self.intentos_acum.max()) + 1)
                    self.ax2.plot(k_vals, n * stats.geom.pmf(k_vals, self.prob_cara),
                                  color=p["red"], linewidth=1.8, label="Geométrica teórica")
                    self.ax2.legend(fontsize=8, frameon=False, labelcolor=p["ink2"])

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
                borde3, ancho3 = self._estilo_borde(bins_chi2)
                self.ax3.hist(chi2_vals, bins=bins_chi2, color=p["aqua"], edgecolor=borde3,
                               linewidth=ancho3)
                x3 = np.linspace(max(chi2_vals.min(), 1e-6), chi2_vals.max(), 200)
                ancho_bin3 = (chi2_vals.max() - chi2_vals.min()) / bins_chi2
                self.ax3.plot(x3, len(chi2_vals) * ancho_bin3 * stats.chi2.pdf(x3, df=k_gl),
                               color=p["red"], linewidth=2, label="χ² teórica")
                self.ax3.legend(fontsize=8, frameon=False, labelcolor=p["ink2"])

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

        self.ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.9, color=self.color_grid())
        self.ax2.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.9, color=self.color_grid())
        self.ax2.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6, color=self.color_grid())
        self.ax3.grid(True, linestyle="--", linewidth=0.7, alpha=0.9, color=self.color_grid())

        self.status.set_text(f"Total simulado: {n:,} repeticiones".replace(",", "."))

        self.aplicar_tema()
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    App()
