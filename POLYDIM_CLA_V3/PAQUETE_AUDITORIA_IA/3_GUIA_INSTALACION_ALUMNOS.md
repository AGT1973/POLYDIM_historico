# POLYDIM V3.2: Guía de Instalación para Estudiantes (Mac / Linux)

Bienvenido a POLYDIM. Dado que este motor requiere compilación dual (PyTorch en Python y el núcleo PMTP en Rust), la instalación puede parecer compleja. Sin embargo, si utilizas **Google Antigravity**, el agente puede encargarse de todo el proceso de compilación y entorno virtual por ti.

---

## Opción 1: Instalación Autónoma con Antigravity (Recomendada)

Si tienes [Antigravity CLI (`agy`)](https://github.com/google/antigravity) instalado en tu Mac o Linux, el proceso es completamente automático.

1. Abre tu terminal y clona este repositorio:
   ```bash
   git clone https://github.com/AGT1973/POLYDIM.git
   cd POLYDIM
   ```

2. Inicia tu sesión de Antigravity en la raíz del proyecto:
   ```bash
   agy
   ```

3. Simplemente indícale al agente tu objetivo usando el comando de sistema `/goal`. Copia y pega el siguiente prompt:
   > `/goal Configura el entorno para POLYDIM V3.2. Por favor: 1) Crea un entorno virtual e instala los requerimientos de Python (PyTorch). 2) Dirígete a Motor_v3/pmtp y compila el núcleo de Rust usando cargo build --release. 3) Ejecuta los tests unitarios en Rust (cargo test) para verificar la instalación.`

Antigravity leerá los archivos de configuración, detectará si te falta alguna dependencia del sistema (como `rustup` o herramientas de compilación de C++ en Mac/Linux) y las instalará interactuando con tu terminal hasta que todo el pipeline esté en verde.

---

## Opción 2: Instalación Manual (Paso a Paso)

Si prefieres realizar el proceso manualmente o no cuentas con Antigravity, sigue estos pasos:

### 1. Prerrequisitos del Sistema
Asegúrate de tener instalados:
- **Python 3.10+**
- **Rust y Cargo:** Instálalos mediante Rustup ejecutando en tu terminal:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  *(Reinicia tu terminal después de la instalación).*

### 2. Entorno Python (Motor Matemático)
Configura tu entorno para el motor de PyTorch:
```bash
# Dentro de la carpeta POLYDIM
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -e .
```

### 3. Compilación del Núcleo Rust (Protocolo PMTP)
El enrutador hiperdimensional está escrito en Rust para garantizar DPI=0 y velocidad máxima.
```bash
cd Motor_v3/pmtp

# Compilar en modo Release (optimizado)
cargo build --release

# Ejecutar los tests de validación isométrica
cargo test
```
Si los tests de Rust pasan (MSE < 1e-20), tu nodo local es matemáticamente perfecto y está listo para comunicarse en alta dimensión.

> [!TIP]
> **Usuarios de Mac (Apple Silicon M1/M2/M3):**
> La compilación en Rust aprovechará automáticamente la arquitectura ARM64. Para PyTorch, asegúrate de utilizar la versión con soporte MPS (Metal Performance Shaders) si deseas aceleración por GPU. Antigravity lo configurará automáticamente en la Opción 1.
