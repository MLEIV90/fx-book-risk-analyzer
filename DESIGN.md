# FX Book Risk Analyzer — Design Spec (v0.3)

> Documento de diseño. Define **qué** tiene que ser la herramienta y **de dónde
> sale cada dato**, antes de escribir código. Para revisión crítica.

---

## 1. Propósito y usuario

**Usuario:** la mesa de Treasury / ALM de un proveedor de hedging de FX.

**Para qué:** producir un **informe de riesgo accionable** sobre el libro propio
del proveedor — el riesgo agregado que queda tras venderles coberturas a los
clientes. No es una calculadora de un trade: es el análisis de una cartera.

**Principio rector:** exactitud o declaración explícita del límite — **nunca un
número inventado disfrazado de real**. Todo dato se observa, se deriva de algo
observado, o se declara abiertamente como aproximación con su impacto.

---

## 2. Inversión del flujo

El centro es el **proveedor**, no el cliente.

1. La mesa define **su libro**: posiciones que crea y se guardan.
2. La app baja el **mercado** (spot, historia, tasas) y **deriva** los parámetros
   (volatilidad, correlaciones). El usuario no los inventa.
3. Produce el **análisis de riesgo** del libro completo.
4. La vista **cliente** es una derivada secundaria (qué cotizamos, qué le cuesta).

---

## 3. Qué define el usuario

**El libro** arranca **vacío**. El usuario **crea posiciones** que se guardan y
alimentan el análisis. Cada posición:
- Par · Dirección (long/short base) · Notional · Plazo

**Parámetros de cartera:**
- Horizonte de riesgo (1 / 10 días) · Confianza (95% / 99%) · Ventana histórica.

Spot, tasas, volatilidad y correlaciones **no se tipean**: se bajan o se derivan.

---

## 4. De dónde sale CADA dato — nosotros vs. una empresa real

Tres niveles de origen:
**Observado** = se baja del mercado · **Derivado** = se calcula a partir de lo
observado · **Supuesto** = se asume y se documenta.

| Dato | Lo que usamos (fuente concreta) | Lo que usaría una empresa | Nivel |
|---|---|---|---|
| Spot | Cierre diario de yfinance | Bloomberg / Refinitiv en tiempo real | Observado |
| Historia de precios | Serie de cierres de yfinance | Mismo feed profesional | Observado |
| Retornos | Cálculo: precio_t / precio_t-1 − 1 | Igual (cálculo estándar) | Derivado |
| Volatilidad histórica | Desvío de los retornos, anualizado | Igual | Derivado |
| Volatilidad GARCH | GARCH(1,1)-t y GJR-GARCH sobre los retornos | Igual, o vol implícita de opciones | Derivado (modelo) |
| Volatilidad implícita | — (fuera de v1) | Superficie de vol de Bloomberg | *Next step* |
| Correlaciones | Matriz + rolling de los retornos | Igual | Derivado |
| Tasas por plazo | Tasas reales de FRED (SOFR/ESTR/SONIA y puntos por plazo) | Curva OIS / depósitos completa de Bloomberg | Observado (term structure simplificada) |

**Lo único que NO es observado ni derivado:** nada queda como "supuesto puro" en
v1 — incluso las tasas son reales (FRED). La simplificación declarada es que no
reconstruimos la curva completa con todos los plazos ni el cross-currency basis;
usamos los puntos de FRED disponibles. En producción, eso vendría de un feed con
la curva OIS completa.

---

## 5. Volatilidad vs. métodos de VaR (dos capas distintas)

Para evitar confusión, se declara explícito:

**Capa 1 — estimación de volatilidad (el σ):**
- Histórica simple (desvío anualizado).
- GARCH(1,1)-t (volatilidad cambiante; captura clustering y colas gordas).
- GJR-GARCH(1,1,1)-t (agrega asimetría / efecto apalancamiento) como comparación.

**Capa 2 — método de VaR (qué se hace con la info):**
- **Paramétrico** — usa σ (capa 1) + normal. Rápido; subestima colas.
- **Histórico** — usa los retornos reales; **no** usa σ. Sin supuesto de forma.
- **Monte Carlo** — simula con σ + correlación (capa 1). Motor cambiable (normal/t).

Relación: GARCH/histórica **alimentan** el paramétrico y el Monte Carlo; el VaR
histórico va por afuera. Se muestran las dos estimaciones de vol y los tres VaR,
y se discute por qué difieren.

---

## 6. El análisis que produce

**A. El libro** — posiciones, exposición neta por moneda, MtM por posición y total.

**B. Riesgo de mercado**
- VaR por tres métodos + Expected Shortfall.
- **Backtesting de Kupiec** (validación del modelo).
- Comparación de métodos y de estimadores de vol (histórica vs GARCH).

**C. Atribución de riesgo (lo que distingue senior de junior)**
- Contribución de cada posición al VaR.
- Contribución de cada factor (par) al riesgo.
- Efecto diversificación: VaR cartera vs. suma de individuales.

**D. Riesgo de tasa** — DV01 por curva; exposición al diferencial.

**E. Liquidez** — simulación de variation margin → colchón de caja.

**F. Stress** — escenarios históricos (Brexit, COVID, SNB) sobre el libro de hoy,
comparados con el VaR.

---

## 7. Visualizaciones (que digan algo)

- Distribución de P&L con VaR y ES.
- Correlaciones rolling en el tiempo.
- Contribución al riesgo por posición (barras).
- Volatilidad histórica vs GARCH en el tiempo.
- Backtesting: excepciones del VaR contra el P&L real.
- Curva de tasas por moneda (FRED).

---

## 8. Arquitectura

Motor puro `fxrisk/` (testeado) separado de la interfaz `app.py`.
Nuevos módulos respecto a hoy: `curves.py` (tasas FRED), `garch.py`,
`attribution.py`, `backtest.py`.

---

## 9. Supuestos y límites (declarados)

- Tasas reales de FRED, pero term structure simplificada (sin curva OIS completa
  ni cross-currency basis).
- Volatilidad histórica/GARCH; implícita queda fuera (sin datos de opciones).
- Datos de precio vía yfinance (prototipo); producción = feed profesional.
- VaR paramétrico asume normalidad; por eso histórico + ES + Kupiec + stress.
- Libro de forwards; opciones se valúan aparte (no entran al VaR en v1).

---

## 10. Fuera de alcance (v1) — listado como next steps

- Volatilidad implícita / superficie de vol.
- Curva OIS completa + cross-currency basis.
- Opciones dentro del VaR de cartera.
- Multi-entidad / consolidación.

---

## 11. Fases de construcción

0. Spec aprobado.
1. Datos reales: spot + historia + retornos + correlaciones (yfinance).
2. Tasas reales por plazo (FRED) → `curves.py`.
3. Volatilidad: histórica + GARCH(1,1)-t + GJR-GARCH → `garch.py`. [DONE, tested]
4. Libro editable (vacío, persistente) + exposición neta + MtM.
5. Riesgo de mercado: VaR/ES + Kupiec + atribución.
6. DV01, liquidez, stress.
7. Visualizaciones.
8. Vista cliente (derivada).
9. README + deploy.
