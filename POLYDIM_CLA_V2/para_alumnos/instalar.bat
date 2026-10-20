@echo off
echo ================================================
echo  INSTALADOR AGY - KIT PARA ALUMNOS
echo ================================================

echo Instalando dependencias Python...
pip install mcp httpx fastmcp

echo.
echo SIGUIENTE PASO: Abre config.json y completa tus API keys
echo Despues copia mcp_config.json a:
echo   C:\Users\%USERNAME%\.gemini\config\mcp_config.json
echo.
echo Listo! Abre Antigravity IDE y empieza a usar AGY.
pause
