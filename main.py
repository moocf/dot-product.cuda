# https://www.kaggle.com/wolfram77/cudaf-dot-product
import os
from IPython.display import FileLink
src="dot-product"
out="{}.txt".format(src)
!printf "" > "$out"
display(FileLink(out))
!echo "" && nvidia-smi && echo ""

# Download program
!rm -rf $src
!git clone https://github.com/cudaf/$src
!echo ""

# Run
!nvcc -std=c++17 -Xcompiler -O3 $src/main.cu
!ulimit -s unlimited && stdbuf --output=L ./a.out 2>&1 | tee -a "$out"
