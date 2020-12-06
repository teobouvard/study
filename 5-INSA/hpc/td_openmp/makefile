build-sequential: system_solver_sequential.cpp
	g++ system_solver_sequential.cpp -O3 -o solve 

run-sequential: build-sequential
	./solve input/single_matrix_256.csv 100 output/validation_matrix_256_100.csv

build-parallel: system_solver_parallel.cpp
	g++ system_solver_parallel.cpp -O3 -fopenmp -o solve_parallel 

run-parallel: build-parallel
	./solve_parallel input/single_matrix_8192.csv 100 output/validation_matrix_8192_100.csv

build-multiple: multiple_system_solver_parallel.cpp
	g++ -std=c++17 -Wall -Wextra multiple_system_solver_parallel.cpp -O3 -fopenmp -o solve_multiple 

run-multiple: build-multiple
	./solve_multiple input/multiple_matrix_128.csv 500 output/validation_multiple_matrix_128_100.csv
