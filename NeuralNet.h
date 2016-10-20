#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "Neuron.h"
#include "NeuralData.h"
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace std;

#define BIAS 1
#define WEIGHT_FACTOR 0.1
#define MOMENTUM 0.6

inline double randFloat() {return rand() / (RAND_MAX + 1.0);}
inline double randomClamped() {return WEIGHT_FACTOR * (randFloat() - randFloat());}

class NeuralNet {
private:
	double bias;
	double momentum;
	int inputCount;
	int outputCount;
	int maxEpochCount;
	double errThre;
	int epochCount;
	double learningRate;
	double accuEpcErr;
	double singleError;
	int batchSize;
	int currentEpochNum;
	std::vector<int> layerSize;
	std::string activationType;

	std::vector<NeuronLayer> hiddenLayers;
	NeuronLayer outputLayer;

	std::vector<double> errorPerEpoch;

	std::vector<std::vector<int>> weightChangePara;
	std::vector<std::vector<double>> weightChangeMat;
	std::vector<std::vector<int>> activationPara;
	std::vector<std::vector<double>> activationMat;
	int trainCorrectCount;
	std::vector<std::vector<double>> errorMat;

public:
	NeuralNet(int input_count,
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
			  std::vector<std::vector<int>> activation_para);
	~NeuralNet();

	bool calculateOutput(std::vector<double>& input, std::vector<double>& output);
	//bool trainingEpoch(std::vector<std::vector<double>>& inputSet, std::vector<std::vector<double>>& targetOutputSet);
	//bool train(std::vector<std::vector<double>>& inputSet, std::vector<std::vector<double>>& targetOutputSet);
	int recognize(std::vector<double>& input);

	bool stocGradDescTrain(NeuralData& data, int flag);
	bool batchTrain(std::vector<std::vector<double>>& inputSet, std::vector<std::vector<double>>& targetOutputSet);
	bool singleBackPropagation(std::vector<double>& input, std::vector<double>& targetOutput);

	double getAccuEpcErr() { return this->accuEpcErr; }
	void setAccuEpcErr(double accu_epc_err) { this->accuEpcErr = accu_epc_err; }

	double getSingleError() { return this->singleError; }
	void setSingleError(double single_error) { this->singleError = single_error; }

	int getEpochCount() { return this->epochCount; }
	void setEpochCount(int epoch_count) { this->epochCount = epoch_count; }

	int getOutputCount() { return this->outputCount; }
	void setOutputCount(int output_count) { this->outputCount = output_count; }

	int getInputCount() { return this->inputCount; }
	void setInputCount(int input_count) { this->inputCount = input_count; }

	int getMaxEpochCount() { return this->maxEpochCount; }
	void setMaxEpochCount(int max_epoch_count) { this->maxEpochCount = max_epoch_count; }

	double getErrThre() { return this->errThre; }
	void setErrThre(double err_thre) { this->errThre = err_thre; }

	double getLearningRate() { return this->learningRate; }
	void setLearningRate(double learning_rate) { this->learningRate = learning_rate; }

protected:
	bool createNetwork();
	void initialize();
	double sigmoid(double input);
	double sigmoidDerivative(double x);
	double hptg(double x);
	double hptgDerivative(double x);
	double relu(double x);
	double reluDerivative(double x);
	double actFun(double x);
	double actDer(double x);
	double getAccuracy(NeuralData& data);
	bool saveWeightChange(int currentEpoch);
	bool writeWeightChange();
	bool trainCorrectCountUpdate(std::vector<double>& outputVec, std::vector<double>& targetOutput);
	bool saveError(double trainAccuracy, double testAccuracy);
	bool writeError();
	bool saveActivation();
	bool writeActivation();
};

#endif
