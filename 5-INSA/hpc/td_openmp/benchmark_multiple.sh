make build-multiple

for schedule in "guided" "static"; do
    for k in $(seq 0 4); do
        n_threads=$((2 ** k))
        for iteration in $(seq 10); do
            for n_steps in 100 500 1000; do
                for i in $(seq 0 4); do
                    matrix_size=$((64 * 2 ** i))
                    export OMP_SCHEDULE=${schedule} OMP_NUM_THREADS=${n_threads} && ./solve_multiple input/multiple_matrix_${matrix_size}.csv $n_steps output/validation_multiple_matrix_${matrix_size}_${n_steps}.csv
                done
            done
        done
    done
done
