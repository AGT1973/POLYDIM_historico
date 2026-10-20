# Guía de instalación — IA gratis para Polidim
*Tiempo total: ~15 minutos. No hace falta tarjeta de crédito en ningún paso.*

---

## PASO 1 — Cuenta en OpenRouter (modelo gratis #1)

1. Ir a: **https://openrouter.ai**
2. Click en **Sign up** → registrarse con email, Google o GitHub (lo que sea más rápido).
3. Una vez adentro, ir directo a: **https://openrouter.ai/settings/keys**
4. Click en **Create Key**.
5. Ponerle un nombre (ej: "polidim-tu-nombre") y click **Create**.
6. **Copiar la key inmediatamente** — empieza con `sk-or-v1-...`. OpenRouter la muestra una sola vez; si se pierde hay que crear otra.
7. Guardarla en un lugar seguro (notas, gestor de contraseñas) — la vas a pegar en Cherry Studio en el Paso 3.

**Límites gratis (sin pagar nada):** 20 solicitudes por minuto, 50 por día. Si en algún momento querés más volumen, cargar $10 una sola vez sube el límite diario a 1000 para siempre — pero no es necesario para arrancar.

**Modelo recomendado para probar:** `deepseek/deepseek-r1:free` o `qwen/qwen3-coder:free` (ambos soportan herramientas/tool calling).

---

## PASO 2 — Cuenta en Google AI Studio / Gemini (modelo gratis #2)

1. Ir a: **https://aistudio.google.com/apikey**
2. Iniciar sesión con una cuenta de Google (cualquier Gmail sirve, no hace falta una cuenta nueva).
3. Click en **Create API key**.
4. Si pide elegir un proyecto de Google Cloud: aceptar la opción por defecto ("Create key in new project") — no requiere facturación.
5. Copiar la key generada — empieza con `AIza...`.
6. Guardarla junto con la de OpenRouter.

**Límites gratis:** según el modelo, entre 10 y 15 solicitudes por minuto y varios cientos por día. Usar **Gemini 2.5 Flash** o **Flash-Lite** — son los modelos del tier gratis (Gemini Pro ya no tiene capa gratuita).

**Importante:** en el tier gratis, Google puede usar tus prompts para entrenar sus modelos. Si van a trabajar con información sensible, mejor usar OpenRouter para eso.

---

## PASO 3 — Instalar Cherry Studio (la app que une todo)

1. Ir a: **https://cherry-ai.com** (o directamente a **https://github.com/CherryHQ/cherry-studio/releases** para bajar el instalador).
2. Descargar la versión para Windows (o Mac/Linux según corresponda).
3. Instalar y abrir la app.
4. Ir a **Configuración (ícono de engranaje) → Model Providers / Proveedores de modelo**.
5. **Agregar OpenRouter:**
   - Buscar "OpenRouter" en la lista de proveedores.
   - Pegar la key del Paso 1 (`sk-or-v1-...`).
   - Guardar.
6. **Agregar Google Gemini:**
   - Buscar "Gemini" o "Google" en la lista de proveedores.
   - Pegar la key del Paso 2 (`AIza...`).
   - Guardar.
7. Volver al chat principal: arriba debería poder elegirse el modelo. Probar primero con uno de OpenRouter (`deepseek/deepseek-r1:free`) y después con Gemini Flash, para confirmar que ambos responden.

---

## PASO 4 — Probar que todo funciona

Escribir en el chat de Cherry Studio, con cada modelo:
```
Decime qué modelo eres y la fecha de hoy.
```
Si responde, esa conexión está lista. Repetir cambiando de modelo en el selector de arriba.

---

## Checklist final
- [ ] Cuenta OpenRouter creada, key copiada y guardada
- [ ] Cuenta Google AI Studio creada, key copiada y guardada
- [ ] Cherry Studio instalado
- [ ] OpenRouter conectado en Cherry Studio
- [ ] Gemini conectado en Cherry Studio
- [ ] Probé un mensaje con cada modelo y respondió

---

*Si algo no responde: revisar que la key se pegó completa (sin espacios al principio/final) y que se guardó con el botón correspondiente antes de cambiar de pantalla.*
