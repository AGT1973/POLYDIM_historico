# Tarea 21 — POLYDIM V2 (Escuadrón 2: Álgebra de Clifford y Rotores)

## 1. Enunciado original (banco maestro)

> "21 (Test 21): Actúa como matemático numérico. Escribe un script que exprese
> restricciones monoidales estrictas y variedades que posean curvaturas
> diferentes, midiendo la preservación geodésica local."

## 2. Nota de reconstrucción

El enunciado, tal como está en el banco, es gramaticalmente incompleto: es el
resultado de combinar dos fragmentos de plantilla ("restricciones monoidales
estrictas" + "variedades que posean curvaturas diferentes, midiendo la
preservación geodésica local") sin verbo que los conecte. No se puede resolver
tal cual está escrito.

Reconstrucción investigable que conserva el rol (matemático numérico) y ambos
fragmentos técnicos reconocibles:

> **Diseña e implementa un experimento numérico que evalúe si existe una
> identificación estrictamente asociativa (coherente en el sentido de
> Mac Lane) entre dos variedades bidimensionales embebidas en R^3
> con curvaturas gaussianas distintas, tal que dicha identificación preserve
> las distancias geodésicas locales entre puntos cercanos. Mide numéricamente
> el error de preservación geodésica en función del radio de la vecindad y de
> la diferencia de curvaturas.**

Importante: el enunciado original **da por sentado que tal preservación es
alcanzable** ("midiendo la preservación geodésica local", como si solo faltara
medirla). Esa premisa no está establecida — de hecho, el Teorema Egregium de
Gauss dice exactamente lo contrario para curvaturas distintas. Por eso este
trabajo no "demuestra" la preservación: la trata como pregunta abierta y la
investiga empíricamente, y su conclusión principal es señalar por qué el
enunciado, tal como estaba planteado, asume algo matemáticamente falso salvo
en el límite trivial (radio → 0 o curvaturas iguales).

## 3. Modelo matemático estándar

Sea C la categoría (monoidal estricta, en el sentido de Mac Lane)
cuyos objetos son parches de coordenadas geodésicas normales (r,θ)
alrededor de un punto base p0 en una superficie Σ ⊂ R^3, y cuyos morfismos son las identificaciones de coordenadas entre
parches. La composición de identificaciones de coordenadas es trivialmente
asociativa (es la composición de la función identidad consigo misma), de modo
que la "restricción monoidal estricta" del enunciado se satisface sin
esfuerzo — **eso no es lo que hay que verificar**.

Lo que realmente hay que verificar es si esa identificación de coordenadas,
al usarse para transportar puntos entre dos superficies Σ_A y
Σ_B con curvaturas gaussianas distintas K_A ≠ K_B, preserva la
métrica intrínseca (distancias geodésicas).

**Herramienta: Teorema de Bertrand–Diguet–Puiseux.** La longitud de una
circunferencia geodésica de radio r alrededor de un punto de curvatura
gaussiana K cumple:

L(r) = 2πr − (π/3) K r^3 + O(r^5) = 2πr (1 − K r^2 / 6) + O(r^5)

Si la identificación de coordenadas (r,θ)_A ↦ (r,θ)_B fuera
una isometría local, entonces L_A(r) = L_B(r) para todo r. Como
L(r) depende de K a orden r^3, esto solo puede cumplirse exactamente si
K_A = K_B (consistente con el **Teorema Egregium**: la curvatura gaussiana
es un invariante intrínseco, no puede cambiar bajo una isometría).

**Superficies elegidas:**
- Σ_A: esfera de radio R, X(u,v) = R(sin u cos v, sin u sin v, cos u), curvatura constante K_A = 1/R^2 > 0.
- Σ_B: paraboloide hiperbólico, X(u,v) = (u, v, a(u^2−v^2)), curvatura en el origen K_B = −4a^2 < 0.

**Métrica de error:** ε(r) = |L_A(r) − L_B(r)|, comparada contra
la predicción teórica 2πr(1 − K r^2/6) para cada superficie.

## 4. Razonamiento

- Se descarta medir directamente "distancia entre dos puntos arbitrarios"
  porque requeriría resolver el problema de valores de frontera geodésico
  (boundary value problem), numéricamente más costoso e inestable que el
  problema de valor inicial (shooting method).
- Se elige en cambio la **circunferencia geodésica** (todas las geodésicas
  que salen de p0 integradas hasta el mismo parámetro de arco r), porque
  se resuelve con un shooting method estándar (problema de valor inicial) y
  tiene una fórmula cerrada de referencia (Bertrand–Diguet–Puiseux) validada
  en la literatura clásica de geometría diferencial — no es una métrica
  inventada.
- Las geodésicas se calculan resolviendo numéricamente la ecuación geodésica
  exacta ẍ^k + Γ^k_ij ẋ^i ẋ^j = 0 con los símbolos
  de Christoffel obtenidos simbólicamente (SymPy) a partir de la primera forma
  fundamental — no se aproxima la geometría, se deriva de la parametrización
  exacta de cada superficie.
- Alternativa descartada: usar el operador `PROJECT`/dualidad de Hodge del
  paper POLYDIM directamente como el "morfismo" a probar. Se prefirió
  restringir a coordenadas geodésicas normales porque es el objeto matemático
  estándar más simple sobre el que el resultado (Teorema Egregium) tiene
  fórmula cerrada verificable, evitando introducir supuestos adicionales no
  necesarios para responder la pregunta del enunciado.

## 5. Desarrollo y código

Ver `geodesic_curvature_test.py` adjunto. Resumen del método:

1. Dada una superficie X(u,v) simbólica, se calculan E,F,G (primera forma
   fundamental), L,M,N (segunda forma fundamental) y la curvatura gaussiana
   K=(LN−M²)/(EG−F²), y los símbolos de Christoffel de la métrica, todo
   simbólicamente con SymPy y luego compilados a funciones numéricas
   (`lambdify`).
2. Para un punto base (u0,v0) y un ángulo θ, se dispara una
   geodésica (shooting method, `scipy.integrate.solve_ivp`) con velocidad
   inicial unitaria (norma 1 respecto a la métrica) en la dirección θ,
   integrada hasta parámetro de arco r.
3. Repitiendo para 48 ángulos equiespaciados se obtiene la circunferencia
   geodésica de radio r; su longitud se aproxima como el perímetro del
   polígono que forman los 48 puntos finales embebidos en R^3.
4. Se compara L(r) numérico contra la predicción de Bertrand–Diguet–Puiseux
   y contra L(r) de la otra superficie, para una sucesión de radios r.

### Resultado de la corrida de sanity-check (interna, no reemplaza la corrida del alumno)

```
K esfera: 0.25   (esperado 1/R^2 = 0.25)
K paraboloide en origen: -1.0   (esperado -4a^2 = -1.0)

 r      L_esfera   BDP_esfera   L_parab    BDP_parab   |L_esf-L_par|
0.050    0.31390    0.31413     0.31406    0.31429   0.00016
0.140    0.87830    0.87893     0.88184    0.88252   0.00354
0.230    1.44092    1.44195     1.45640    1.45787   0.01548
0.320    2.00062    2.00204     2.04147    2.04493   0.04085
0.410    2.55627    2.55806     2.64003    2.64828   0.08376
0.500    3.10675    3.10887     3.25421    3.27249   0.14746
```

Ambas curvaturas numéricas coinciden con el valor analítico exacto (chequeo
de correctitud del método). La discrepancia entre L_esfera(r) y
L_parab(r) es prácticamente nula para r pequeño y crece
monótonamente con r, en línea con el término de orden r^3 de
Bertrand–Diguet–Puiseux.

## 6. Conclusión

**El enunciado original, tal como estaba planteado, es matemáticamente
insostenible fuera de un límite trivial.** No existe una identificación de
coordenadas entre dos variedades de curvatura gaussiana distinta que preserve
exactamente las distancias (o circunferencias) geodésicas locales para todo
radio — esto es exactamente lo que prohíbe el Teorema Egregium de Gauss, y la
corrida numérica lo confirma: el error de preservación ε(r) crece
de forma consistente con la predicción teórica de orden O(K r^3) en vez de
mantenerse en cero.

Lo que sí se preserva de forma estricta (categorialmente) es la asociatividad
de la composición de identificaciones de coordenadas — esa parte del
enunciado ("restricciones monoidales estrictas") es trivialmente cierta,
pero **no implica** preservación geométrica. Este es el resultado científico
válido de la tarea: mostrar que la coherencia categórica (Mac Lane) y la
preservación geodésica (invariante de Gauss) son propiedades independientes,
y que la segunda solo se cumple aproximadamente, en el límite r→0 o
cuando las curvaturas coinciden — consistente con el marco de "Homotopía
Aproximada" del Apéndice V2.0 del paper POLYDIM (tolerancia ε en vez
de igualdad estricta).

## 7. Instrucciones para correr el código vos mismo

```bash
pip install sympy scipy numpy --break-system-packages   # si hace falta
python3 geodesic_curvature_test.py
```

No hace falta ninguna otra dependencia. El script corre en menos de un
minuto. Si querés, cambiá `R` (radio de la esfera) o `a` (parámetro del
paraboloide) para ver cómo cambia la velocidad de divergencia de
ε(r) con |K_A − K_B| — es una buena forma de comprobar con tus
propios ojos que la discrepancia escala con la diferencia de curvaturas, tal
como predice la fórmula.

## 8. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 21,
  "tipo_problema": "geometria_diferencial / teoria_de_categorias_monoidales",
  "estado": "resuelto",
  "resultado_resumen": "Se demuestra numericamente (via el teorema de Bertrand-Diguet-Puiseux) que ninguna identificacion de coordenadas entre variedades de curvatura gaussiana distinta preserva distancias geodesicas exactamente, confirmando el Teorema Egregium; la coherencia monoidal estricta (Mac Lane) se cumple pero es independiente de esa preservacion geometrica.",
  "metrica_estandar_usada": "Longitud de circunferencia geodesica L(r) vs prediccion de Bertrand-Diguet-Puiseux L(r) = 2*pi*r*(1 - K*r^2/6)"
}
```
