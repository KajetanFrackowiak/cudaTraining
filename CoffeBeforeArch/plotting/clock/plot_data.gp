set terminal pngcairo enhanced font 'Verdana, 10'
set output 'output.png'

set title "Block Clocks"
set xlabel "Block"
set ylabel "Clocks"
set grid

plot 'output.dat' using 1:2 smooth csplines title 'Block Clocks', 