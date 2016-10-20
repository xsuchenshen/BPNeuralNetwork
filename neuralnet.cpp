#include "NeuralNet.h"

NeuralNet::NeuralNet(int input_count,
					 int output_count,
					 std::vector<int> layer_size,
				 	 double p_bias,
				 	 double p_momentum,
					 double learning_rate,
				 	 int max_epoch_count,
				 	 double err_thre,
				     int batch_size,
				 	 std::string activation_type,
					 std::vector<std::vector<int>> weight_change_para,
	   			  	 std::vector<std::vector<int>> activation_para) {

	this->inputCount = input_count;
	this->outputCount = output_count;
	this->bias = p_bias;
	this->momentum = p_momentum;
	this->learningRate = learning_rate;
	this->maxEpochCount = max_epoch_count;
	this->errThre = err_thre;
	this->layerSize = layer_size;
	this->batchSize = batch_size;
	this->activationType = activation_type;
	this->weightChangePara = weight_change_para;
	this->activationPara = activation_para;

	this->hiddenLayers.resize(layer_size.size() - 2);

	this->createNetwork();
	this->initialize();
}


NeuralNet::~NeuralNet() {

}


bool NeuralNet::createNetwork() {
	int i;
	for (i = 1; i < this->layerSize.size() - 1; i++) {
		if (i == 1) {
			this->hiddenLayers[i - 1].initialize(this->layerSize[i], this->inputCount);
		} else {
			this->hiddenLayers[i - 1].initialize(this->layerSize[i], this->layerSize[i - 1]);
		}
	}

	this->outputLayer.initialize(this->layerSize[i], this->layerSize[i - 1]);

	return true;
}


void NeuralNet::initialize() {
	for (int i = 0; i < this->hiddenLayers.size(); i++) {
		for (int j = 0; j < this->hiddenLayers[i].neuronCount; j++) {
			for (int k = 0; k < this->hiddenLayers[i].neurons[j].inputCount; k++) {
				this->hiddenLayers[i].neurons[j].weights[k] = randomClamped();
				this->hiddenLayers[i].neurons[j].lastWeightUpdate[k] = 0.0;
			}
		}
	}

	for (int i = 0; i < this->outputLayer.neuronCount; i++) {
		for (int j = 0; j < this->outputLayer.neurons[i].inputCount; j++) {
			this->outputLayer.neurons[i].weights[j] = randomClamped();
			this->outputLayer.neurons[i].lastWeightUpdate[j] = 0.0;
		}
	}

	this->accuEpcErr = 0.0;
	this->epochCount = 0;
	this->trainCorrectCount = 0;
	this->currentEpochNum = 0;
}


bool NeuralNet::calculateOutput(std::vector<double>& input, std::vector<double>& output) {
	int i, j, k;

	double inputSum;

	for (i = 0; i < this->hiddenLayers.size(); i++) {
		for (j = 0; j < this->hiddenLayers[i].neuronCount; j++) {
			inputSum = 0.0;
			for (k = 0; k < this->hiddenLayers[i].neurons[j].inputCount - 1; k++) {
				inputSum += this->hiddenLayers[i].neurons[j].weights[k]
							* ((i == 0)? input[k] : this->hiddenLayers[i - 1].neurons[k].activation);
			}
			inputSum += this->hiddenLayers[i].neurons[j].weights[k] * this->bias;
			this->hiddenLayers[i].neurons[j].activation = this->actFun(inputSum);
		}
	}

	for (i = 0; i < this->outputLayer.neuronCount; i++) {
		inputSum = 0.0;
		for (j = 0; j < this->outputLayer.neurons[i].inputCount - 1; j++) {
			inputSum += this->outputLayer.neurons[i].weights[j]
						* this->hiddenLayers.back().neurons[j].activation;
		}
		inputSum += this->outputLayer.neurons[i].weights[j] * this->bias;
		this->outputLayer.neurons[i].activation = this->actFun(inputSum);

		output.push_back(this->outputLayer.neurons[i].activation);
	}


	return true;
}

bool NeuralNet::singleBackPropagation(std::vector<double>& input, std::vector<double>& targetOutput) {
	std::vector<double> outputVec;
	int i, j, k;
	double error;

	if (!this->calculateOutput(input, outputVec)) {
		std::cout << "ERROR CALCULATING OUTPUT\n";
		return false;
	}
	this->trainCorrectCountUpdate(outputVec, targetOutput);
	this->saveActivation();

	// output layer
	for (i = 0; i < this->outputLayer.neuronCount; i++) {
		error = (targetOutput[i] - outputVec[i]) * this->actDer(outputVec[i]);
		this->outputLayer.neurons[i].error = error;
		this->accuEpcErr += std::abs(targetOutput[i] - outputVec[i]);

		for (j = 0; j < this->outputLayer.neurons[i].inputCount - 1; j++) {
			this->outputLayer.neurons[i].dw[j] += error * this->hiddenLayers.back().neurons[j].activation;
		}
		this->outputLayer.neurons[i].db += error * this->bias;
	}

	// hidden layers
	for (i = this->hiddenLayers.size() - 1; i > -1; i--) {
		for (j = 0; j < this->hiddenLayers[i].neuronCount; j++) {
			error = 0.0;
			if (i == this->hiddenLayers.size() - 1) {
				for (k = 0; k < this->outputLayer.neuronCount; k++) {
					error += this->outputLayer.neurons[k].error * this->outputLayer.neurons[k].weights[j];
				}
			} else {
				for (k = 0; k < this->hiddenLayers[i + 1].neuronCount; k++) {
					error += this->hiddenLayers[i + 1].neurons[k].error * this->hiddenLayers[i + 1].neurons[k].weights[j];
				}
			}
			error *= this->actDer(this->hiddenLayers[i].neurons[j].activation);
			this->hiddenLayers[i].neurons[j].error = error;

			if (i == 0) {
				for (k = 0; k < this->hiddenLayers[i].neurons[j].inputCount - 1; k++) {
					this->hiddenLayers[i].neurons[j].dw[k] += error * input[k];
				}
			} else {
				for (k = 0; k < this->hiddenLayers[i].neurons[j].inputCount - 1; k++) {
					this->hiddenLayers[i].neurons[j].dw[k] += error * this->hiddenLayers[i - 1].neurons[k].activation;
				}
			}
			this->hiddenLayers[i].neurons[j].db += error * bias;
		}
	}

	return true;
}

bool NeuralNet::batchTrain(std::vector<std::vector<double>>& inputSet, std::vector<std::vector<double>>& targetOutputSet) {
	int i, j, k;
	double delta;
	for (int i = 0; i < inputSet.size(); i++) {
		this->singleBackPropagation(inputSet[i], targetOutputSet[i]);
	}
	// hidden layers
	for (i = 0; i < this->hiddenLayers.size(); i++) {
		for (j = 0; j < this->hiddenLayers[i].neuronCount; j++) {
			for (k = 0; k < this->hiddenLayers[i].neurons[j].inputCount - 1; k++) {
				delta = (this->learningRate / inputSet.size()) * this->hiddenLayers[i].neurons[j].dw[k]
						+ this->hiddenLayers[i].neurons[j].lastWeightUpdate[k] * this->momentum;
				this->hiddenLayers[i].neurons[j].weights[k] += delta;
				this->hiddenLayers[i].neurons[j].lastWeightUpdate[k] = delta;
				this->hiddenLayers[i].neurons[j].dw[k] = 0.0;
			}
			delta = (this->learningRate / inputSet.size()) * this->hiddenLayers[i].neurons[j].db
					+ this->hiddenLayers[i].neurons[j].lastWeightUpdate[k] * this->momentum;
			this->hiddenLayers[i].neurons[j].weights[k] += delta;
			this->hiddenLayers[i].neurons[j].lastWeightUpdate[k] = delta;
			this->hiddenLayers[i].neurons[j].db = 0.0;
		}
	}

	// output layer
	for (j = 0; j < this->outputLayer.neuronCount; j++) {
		for (k = 0; k < this->outputLayer.neurons[j].inputCount - 1; k++) {
			delta = (this->learningRate / inputSet.size()) * this->outputLayer.neurons[j].dw[k]
					+ this->outputLayer.neurons[j].lastWeightUpdate[k] * this->momentum;
			this->outputLayer.neurons[j].weights[k] += delta;
			this->outputLayer.neurons[j].lastWeightUpdate[k] = delta;
			this->outputLayer.neurons[j].dw[k] = 0.0;
		}
		delta = (this->learningRate / inputSet.size()) * this->outputLayer.neurons[j].db
				+ this->outputLayer.neurons[j].lastWeightUpdate[k] * this->momentum;
		this->outputLayer.neurons[j].weights[k] += delta;
		this->outputLayer.neurons[j].lastWeightUpdate[k] = delta;
		this->outputLayer.neurons[j].db = 0.0;
	}

	return true;
}

bool NeuralNet::stocGradDescTrain(NeuralData& data, int flag) {
	int currEpcNum = 1;
	double accuracy;
	do {
		if (currEpcNum > this->maxEpochCount) {
			break;
		}
		this->currentEpochNum = currEpcNum;

		for (int i = 0; i < data.imgMat.size(); i += this->batchSize) {
			std::vector<std::vector<double>> batchInput(data.imgMat.begin() + i, data.imgMat.begin() + i + this->batchSize);
			std::vector<std::vector<double>> batchTargetOuput(data.tgtMat.begin() + i, data.tgtMat.begin() + i + this->batchSize);

			if (!this->batchTrain(batchInput, batchTargetOuput)) {
				std::cout << "ERROR TRAINING EPOCH\n";
				return false;
			}
		}

		// save weight change
		this->saveWeightChange(currEpcNum);

		//
		if (flag == 1) {
			accuracy = this->getAccuracy(data);
		}

		double trainAccuracy = (double)this->trainCorrectCount / (double)data.imgMat.size();
		//this->singleError = this->accuEpcErr / (this->outputCount * data.imgMat.size());
		//this->accuEpcErr = 0.0;
		this->trainCorrectCount = 0;
		// if (this->singleError < this->errThre) {
		// 	break;
		// }
		//this->errorPerEpoch.push_back(this->singleError);
		std::cout << "Epoch = " << currEpcNum << " Test Accuracy = " << accuracy << " Training Accuracy = " << trainAccuracy << " " << std::flush << "\n";
		this->saveError(trainAccuracy, accuracy);

		currEpcNum++;
	} while (1);
	std::cout << "\n";
	this->writeWeightChange();
	this->writeError();
	this->writeActivation();

	return true;
}

int NeuralNet::recognize(std::vector<double>& input) {
	int matchedClass;
	int idx;
	std::vector<double>::iterator result;

	std::vector<double> output;
	if (!this->calculateOutput(input, output)) {
		return -1;
	}

	result = std::max_element(output.begin(), output.end());
	idx = std::distance(output.begin(), result);

	return (int)idx;
}

double NeuralNet::getAccuracy(NeuralData& data) {
	int count = 0;
	int rec;
	for (int i = 0; i < data.imgMatTest.size(); i++) {
		rec = this->recognize(data.imgMatTest[i]);
		if (rec == data.tgtVecTest[i]) {
			count++;
		}
	}

	return ((double)count / (double)(data.imgMatTest.size()));
}

bool NeuralNet::saveWeightChange(int currentEpoch) {
	int i, j, k;
	for (i = 0; i < this->weightChangePara.size(); i++) {
		std::vector<double> tmp = {(double)currentEpoch, (double)(this->weightChangePara[i][0]), (double)(this->weightChangePara[i][1])};
		if (this->weightChangePara[i][0] == this->layerSize.size() - 1) {
			for (k = 0; k < this->outputLayer.neurons[this->weightChangePara[i][1]].inputCount; k++) {
				tmp.push_back(this->outputLayer.neurons[this->weightChangePara[i][1]].weights[k]);
			}
		} else {
			for (k = 0; k < this->hiddenLayers[this->weightChangePara[i][0] - 1].neurons[this->weightChangePara[i][1]].inputCount; k++) {
				tmp.push_back(this->hiddenLayers[this->weightChangePara[i][0] - 1].neurons[this->weightChangePara[i][1]].weights[k]);
			}
		}
		this->weightChangeMat.push_back(tmp);
	}

	return true;
}

bool NeuralNet::writeWeightChange() {
	std::string weightChangePath = "../data/WEIGHTCHANGE.txt";
	//std::string activationChangePath = "../data/ACTIVATIONCHANGE.txt";
	std::ofstream out(weightChangePath, ios::trunc);
	for (int i = 0; i < this->weightChangeMat.size(); i++) {
		for (int j = 0 ; j < this->weightChangeMat[i].size(); j++) {
			out << this->weightChangeMat[i][j] << " ";
		}
		out << "\n";
	}

	return true;
}

bool NeuralNet::trainCorrectCountUpdate(std::vector<double>& outputVec, std::vector<double>& targetOutput) {
	int idx1, idx2;
	std::vector<double>::iterator result;

	result = std::max_element(outputVec.begin(), outputVec.end());
	idx1 = std::distance(outputVec.begin(), result);
	result = std::max_element(targetOutput.begin(), targetOutput.end());
	idx2 = std::distance(targetOutput.begin(), result);

	if ((int)idx1 == (int)idx2) {
		this->trainCorrectCount++;
	}

	return true;
}

bool NeuralNet::saveError(double trainAccuracy, double testAccuracy) {
	std::vector<double> tmp = {(1.0 - testAccuracy), (1.0 - trainAccuracy)};
	this->errorMat.push_back(tmp);

	return true;
}

bool NeuralNet::writeError() {
	std::string errorPath = "../data/ERRORCHANGE.txt";
	std::ofstream out(errorPath, ios::trunc);

	for (int i = 0; i < this->errorMat.size(); i++) {
		for (int j = 0 ; j < this->errorMat[i].size(); j++) {
			out << this->errorMat[i][j] << " ";
		}
		out << "\n";
	}

	return true;
}

bool NeuralNet::saveActivation() {
	std::vector<double> tmp = {0.0};
	int indicator = 0;
	for (int i = 0; i < this->activationPara.size(); i++) {
		if (this->currentEpochNum == this->activationPara[i][0]) {
			indicator++;//cout<<this->currentEpochNum<<" "<<this->activationPara[i][0];
			//std::vector<double> tmp = {(double)this->currentEpoch};
			if (this->activationPara[i][1] == this->layerSize.size() - 1) {
				tmp[0] = (double)(this->currentEpochNum);
				tmp.push_back((double)(this->activationPara[i][1]));
				tmp.push_back((double)(this->activationPara[i][2]));
				tmp.push_back(this->outputLayer.neurons[this->activationPara[i][2]].activation);

			} else {
				tmp[0] = (double)(this->currentEpochNum);
				tmp.push_back((double)(this->activationPara[i][1]));
				tmp.push_back((double)(this->activationPara[i][2]));
				tmp.push_back(this->hiddenLayers[this->activationPara[i][1] - 1].neurons[this->activationPara[i][2]].activation);

			}

		}
	}
	if (indicator > 0) {
		this->activationMat.push_back(tmp);
	}


	// if (this->currentEpochNum == epochNum) {
	// 	int i, j, k;
	// 	std::vector<double> tmp;
	//
	// 	for (i = 0; i < this->hiddenLayers.size(); i++) {
	// 		tmp.push_back(this->hiddenLayers[i].neurons[4].activation);
	// 		tmp.push_back(this->hiddenLayers[i].neurons[9].activation);
	// 	}
	// 	tmp.push_back(this->outputLayer.neurons[4].activation);
	// 	tmp.push_back(this->outputLayer.neurons[7].activation);
	// 	this->activationMat.push_back(tmp);
	// }

	return true;
}

bool NeuralNet::writeActivation() {
	std::string activationChangePath = "../data/ACTIVATIONCHANGE.txt";
	std::ofstream out(activationChangePath, ios::trunc);

	for (int i = 0; i < this->activationMat.size(); i++) {
		for (int j = 0 ; j < this->activationMat[i].size(); j++) {
			out << this->activationMat[i][j] << " ";
		}
		out << "\n";
	}

	return true;
}

double NeuralNet::sigmoid(double input) {
	double response = 1.0;

	return (1.0 / (1.0 + exp((-1.0) * input / response)));
}

double NeuralNet::sigmoidDerivative(double x) {
	return (x * (1.0 - x));
}

double NeuralNet::hptg(double x) {
	return ((exp(x) - exp((-1.0) * x)) / (exp(x) + exp((-1.0) * x)));
}

double NeuralNet::hptgDerivative(double x) {
	return (1.0 - x * x);
}

double NeuralNet::relu(double x) {
	return (log(1.0 + exp(x)));
}

double NeuralNet::reluDerivative(double x) {
	return ((exp(x) - 1.0) / exp(x));
}

double NeuralNet::actFun(double x) {
	if (this->activationType == "sigmoid") {
		return this->sigmoid(x);
	} else if (this->activationType == "hyperbolic") {
		return this->hptg(x);
	} else if (this->activationType == "relu") {
		return this->relu(x);
	} else {
		return this->sigmoid(x);
	}
}

double NeuralNet::actDer(double x) {
	if (this->activationType == "sigmoid") {
		return this->sigmoidDerivative(x);
	} else if (this->activationType == "hyperbolic") {
		return this->hptgDerivative(x);
	} else if (this->activationType == "relu") {
		return this->reluDerivative(x);
	} else {
		return this->sigmoidDerivative(x);
	}
}
