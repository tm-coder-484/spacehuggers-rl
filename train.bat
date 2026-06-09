@echo off
REM ============================================================================
REM  train.bat -- SpaceHuggers PPO training launcher
REM
REM  Default (no preset):
REM    train [extra flags]            12 envs, no pin   (last-wins on overrides)
REM
REM  Presets -- pick a config with --p N:
REM    train --p 1                     6 envs,  E-core pin
REM    train --p 2                     6 envs,  P-core pin
REM    train --p 3                    12 envs,  no pin
REM    train --p 4                    12 envs,  E-core pin
REM
REM  Extra args after the preset number are passed through:
REM    train --p 2 --repeat 5         preset 2 but override --repeat
REM ============================================================================

REM ---- Cheat sheet (printed every run) ---------------------------------------
echo ============================================================
echo  train --p 1     ^<- 6 envs,  E-core pin
echo  train --p 2     ^<- 6 envs,  P-core pin
echo  train --p 3     ^<- 12 envs, no pin
echo  train --p 4     ^<- 12 envs, E-core pin
echo  train           ^<- default: 12 envs, no pin
echo  train --pin p   ^<- default but pinned to P (direct form)
echo.
echo  add --v to ANY of the above for a live temp/clock monitor window
echo    e.g.  train --p 4 --v      train --v
echo  add --v2 for Tier-2: richer obs + [512,768,512] net (fresh, separate)
echo    e.g.  train --p 3 --v2     (needs node-workers; uses game_models_v2/)
echo ============================================================
echo.

pushd "%~dp0"

if /I not "%~1"=="--p" goto default

if "%~2"=="1" goto p1
if "%~2"=="2" goto p2
if "%~2"=="3" goto p3
if "%~2"=="4" goto p4
goto unknown

:default
python -u train_game.py --forever --backend node-workers --envs 12 --repeat 3 %*
goto end

:p1
echo [preset 1]  6 envs, E-core pin
python -u train_game.py --forever --backend node-workers --envs 6 --repeat 3 --pin e %3 %4 %5 %6 %7 %8 %9
goto end

:p2
echo [preset 2]  6 envs, P-core pin
python -u train_game.py --forever --backend node-workers --envs 6 --repeat 3 --pin p %3 %4 %5 %6 %7 %8 %9
goto end

:p3
echo [preset 3]  12 envs, no pin
python -u train_game.py --forever --backend node-workers --envs 12 --repeat 3 --pin none %3 %4 %5 %6 %7 %8 %9
goto end

:p4
echo [preset 4]  12 envs, E-core pin
python -u train_game.py --forever --backend node-workers --envs 12 --repeat 3 --pin e %3 %4 %5 %6 %7 %8 %9
goto end

:unknown
echo Unknown preset: %~2
echo.
echo Available presets:
echo   train --p 1     6 envs,  E-core pin
echo   train --p 2     6 envs,  P-core pin
echo   train --p 3    12 envs,  no pin
echo   train --p 4    12 envs,  E-core pin
echo.
echo Or run "train" with no preset for the default (12 envs, no pin).
popd
exit /b 1

:end
popd
