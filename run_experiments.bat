@echo off
set VAR_2=1
:loop
"<path_to_environment>\python.exe" "<path_to_repo>\Hyperspectral_GP.py" %VAR_2% "sediment_exp8.yml"
set /A VAR_2 = VAR_2 + 1
if %VAR_2% neq 6 goto loop
pause