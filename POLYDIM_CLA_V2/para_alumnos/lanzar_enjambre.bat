@echo off
echo ========================================================
echo  🚀 INICIANDO ENJAMBRE AGY - PROTOCOLO DELTA Y EPSILON
echo ========================================================
echo.
echo Ejecutando Antigravity en modo silencioso (Telemetry on).
echo Las trazas de error y fallos de modelos de IA iran a:
echo .agents/logs/orchestrator_telemetry.log
echo.

if not exist ".agents\logs" mkdir ".agents\logs"

:: Ejecucion de Antigravity con los flags de orquestacion pura
agy instantiate project --config .agents\AGENTS.md --rules .agents\SOUL_AGY_ORCHESTRATOR.md --silent-retries --log-file .agents\logs\orchestrator_telemetry.log

pause