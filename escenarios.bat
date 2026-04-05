@echo off
setlocal
REM Ajusta el nombre del entorno si hace falta
call conda activate CODE

set OUT=scenarios

echo === (12) few-loose ===
for /L %%S in (1,1,12) do (
  python scenario_maker.py --preset few-loose --seed %%S --out %OUT%
)

echo === (12) few-tight ===
for /L %%S in (1,1,12) do (
  python scenario_maker.py --preset few-tight --seed %%S --out %OUT%
)

echo === (12) many-medium ===
for /L %%S in (1,1,12) do (
  python scenario_maker.py --preset many-medium --seed %%S --out %OUT%
)

echo === (8) heavy-tight ===
for /L %%S in (1,1,8) do (
  python scenario_maker.py --preset heavy-tight --seed %%S --out %OUT%
)

echo === (6) custom overrides ===
for %%N in (10 12 14 16 18 20) do (
  python scenario_maker.py --preset many-medium --seed %%N --n-planes %%N --p-target 3 --slack tight --out %OUT% --name scn_custom_many_tight_pl%%N
)

echo Hecho. Escenarios en: %OUT%
pause
