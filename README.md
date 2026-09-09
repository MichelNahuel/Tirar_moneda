# Tirar moneda

Simulación interactiva (matplotlib) de tres experimentos con monedas, donde se
van agregando repeticiones a voluntad y se ve cómo se forman las distribuciones.

1. **Binomial → Normal**: lanzar 100 monedas, contar caras, repetir. El histograma
   de conteos tiende a una normal (Teorema Central del Límite).
2. **Geométrica → cola larga**: lanzar hasta que sale cara, anotar cuántos lanzamientos
   hicieron falta. La distribución decae rápido y tiene cola larga; en log-log se ve
   casi lineal.
3. **Chi-cuadrado**: estandarizar cada conteo de caras (`Z = (X − 50) / 5`), agrupar en
   bloques de `k` y sumar los cuadrados → distribución chi-cuadrado con `k` grados de libertad.

## Uso

```bash
pip install -r requirements.txt
python Tirar_moneda.py
```

Se abre una ventana con controles (cuadros de texto, botones y radio buttons) para
elegir el experimento y sumar repeticiones.
