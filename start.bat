@echo off
setlocal

echo Select startup mode:
echo 1. Push specific brawler
echo 2. Push all brawlers to trophy threshold
set "PYLA_STARTUP_MODE=single"
set "PYLA_ALL_BRAWLERS_THRESHOLD=1000"
set /p "PYLA_MODE_CHOICE=Enter 1 or 2 [1]: "

if "%PYLA_MODE_CHOICE%"=="2" (
    set "PYLA_STARTUP_MODE=all_to_threshold"
    set /p "PYLA_ALL_BRAWLERS_THRESHOLD=Enter target trophy threshold [1000]: "
    if not defined PYLA_ALL_BRAWLERS_THRESHOLD set "PYLA_ALL_BRAWLERS_THRESHOLD=1000"
)

pyla_main.exe
pause
