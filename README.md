# Taxi-Routing-Hopcroft-Karp
Course project for CS6023: GPU Programming

An CUDA implementation of the method mentioned [here](https://www.nature.com/articles/s41586-018-0095-1)
Parallel algorithm for Hopcroft-Karp from [here](https://hal.inria.fr/hal-00923449/document)

####Files
1.	sequentialcode.cpp
	*	Contains a serial implementation of Hopcroft-Karp for comparison
2.	main.cu 
	*	Contains the main CUDA code
3.	test.cu
	*	Contains a version of the algorithm which uses a primitive version of dynamic parallelism
4.	data_.txt
	*	These text files contain the processed dataset used to run the code
5.	tripdata.csv
	*	Base dataset used for generating data.txt
6.	locations.csv
	*	Location coordinates which are used in generating data.txt 