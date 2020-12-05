for iteration in $(seq 10); do
    for n_steps in 100 500 1000 5000 10000 50000; do
        for i in {1..7}; do
            matrix_size=$((64 * 2 ** i))
            ./solve input/single_matrix_${matrix_size}.csv $n_steps output/validation_matrix_${matrix_size}_${n_steps}.csv
        done
    done
done
