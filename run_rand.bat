@echo off

g++ ./code_and_files/cpp_code/kmeans.cpp -o kmeans.exe

REM ===== checks to see if the code was compiled successfully with g++

if %ERRORLEVEL%==1 (
    echo "ERROR :: code did not compile | compiler is g++"
    exit
)


REM ===== checking for the right amount of arguments
if "%5"=="" (
    echo "ERROR :: there must be exactly 5 parameters"
    echo "ARGS  :: file.txt , K_value , iterations , convergence , total runs"
    exit
)
if not "%6"=="" (
    echo "ERROR :: Too many args provided | program takes exactly 5 parameters"
    exit
)

REM ===== running compiled code 
kmeans %1 %2 %3 %4 %5

pause