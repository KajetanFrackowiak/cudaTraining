set title "Matrix Multiplication Execution Time Comparison"
set xlabel "Matrix Size (N)"
set ylabel "Execution Time (ms)"
set grid

plot "naive_mmul.dat" using 1 with linespoints title "Naive Matrix Multiplication", \
     "aligned_mmul.dat" using 1 with linespoints title "Aligned Matrix Multiplication", \
     "tiled_mmul.dat" using 1 with linespoints title "Tiled Matrix Multiplication"
