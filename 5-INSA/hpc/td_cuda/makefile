ex1:
	nvcc cuda_td_1.cu -o bin/td_1 && ./bin/td_1 dataset/vector/input0.raw dataset/vector/input1.raw dataset/vector/output.raw

ex2:
	nvcc cuda_td_2.cu -o bin/td_2 && ./bin/td_2 dataset/images/logo-insa-lyon-p6.ppm dataset/images/logo-insa-lyon-bw.ppm

ex3:
	nvcc -Xcompiler -Wall -Xcompiler -Werror cuda_td_3.cu -o bin/td_3 && ./bin/td_3 dataset/images/logo-insa-lyon-p6.ppm dataset/images/logo-insa-lyon-blurred.ppm

ex4:
	nvcc -Xcompiler -Wall -Xcompiler -Werror cuda_td_4.cu -o bin/td_4 && ./bin/td_4 dataset/matrix/input0.raw dataset/matrix/input1.raw dataset/matrix/output.raw

ex5:
	nvcc -Xcompiler -Wall -Xcompiler -Werror cuda_td_5.cu -o bin/td_5 && ./bin/td_5 dataset/matrix/input0.raw dataset/matrix/input1.raw dataset/matrix/output.raw