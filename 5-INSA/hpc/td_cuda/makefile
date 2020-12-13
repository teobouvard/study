ex1:
	nvcc cuda_td_1.cu -o bin/td_1 && ./bin/td_1 dataset/vector/input0.raw dataset/vector/input1.raw dataset/vector/output.raw

ex2:
	nvcc cuda_td_2.cu -o bin/td_2 && ./bin/td_2 dataset/images/logo-insa-lyon-p6.ppm dataset/images/logo-insa-lyon-bw.ppm

ex3:
	nvcc -Xcompiler -Wall -Xcompiler -Werror cuda_td_3.cu -o bin/td_3 && ./bin/td_3 dataset/images/logo-insa-lyon-p6.ppm dataset/images/logo-insa-lyon-blurred.ppm