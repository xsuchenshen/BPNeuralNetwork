#ifndef __NEURON_H__
#define __NEURON_H__

#include <vector>
#include <string>

using namespace std;

struct Neuron {
	int inputCount;
	std::vector<double> weights;
	std::vector<double> lastWeightUpdate;
	double activation;
	double error;
	std::vector<double> dw;
	double db;

	void initialize(int input_count) {
		this->inputCount = input_count + 1;
		this->weights.resize(this->inputCount);
		this->lastWeightUpdate.resize(this->inputCount);
		this->activation = 0.0;
		this->error = 0.0;

		this->dw.resize(input_count, 0.0);
		this->db = 0.0;
	}

	~Neuron() {

	}

};

struct NeuronLayer {
	int neuronCount;
	std::vector<Neuron> neurons;

	void initialize(int neuron_count, int input_count_per_neuron) {
		this->neuronCount = neuron_count;
		this->neurons.resize(neuron_count);

		for (int i = 0; i < neuron_count; i++) {
			this->neurons[i].initialize(input_count_per_neuron);
		}
	}

	~NeuronLayer() {

	}
};

struct NeuralParameter {
	int inputLayerCount;
	int outputLayerCount;
	int hiddenLayerCount;
	int neuCntPerHidLyr;
	int epochCount;
};

#endif
