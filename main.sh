#!/usr/bin/env bash
src="dot-product"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/cudaf/$src
cd $src

# Run
nvcc -std=c++17 -Xcompiler -O3 main.cu
stdbuf --output=L ./a.out 2>&1 | tee -a "$out"
