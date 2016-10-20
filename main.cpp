#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "NeuralNet.h"
#include "NeuralData.h"
#include "Utils.h"

using namespace std;

bool loadParameters(std::string& training_data,
					std::string& training_labels,
					std::string& test_data,
					std::string& test_labels,
					int& input_count,
					int& output_count,
					std::vector<int>& layer_size,
					double& p_bias,
					double& p_momentum,
					double& learning_rate,
					int& max_epoch_count,
					double& err_thre,
					int& batch_size,
					std::string& activation_type,
					std::vector<std::vector<int>>& weight_change_para,
					std::vector<std::vector<int>>& activation_para);

int main(int argc, char** argv) {
	std::string training_data;
	std::string training_labels;
	std::string test_data;
	std::string test_labels;
	int input_count;
	int output_count;
	std::vector<int> layer_size;
	double p_bias;
	double p_momentum;
	double learning_rate;
	int max_epoch_count;
	double err_thre;
	int batch_size;
	std::string activation_type;
	std::vector<std::vector<int>> weight_change_para;
	std::vector<std::vector<int>> activation_para;

	loadParameters(training_data,
				   training_labels,
				   test_data,
				   test_labels,
				   input_count,
				   output_count,
			   	   layer_size,
			   	   p_bias,
			   	   p_momentum,
			   	   learning_rate,
			   	   max_epoch_count,
			       err_thre,
			       batch_size,
			       activation_type,
			   	   weight_change_para,
			   	   activation_para);



	NeuralData data;
	data.loadTrainingData(training_data, training_labels);
	data.loadTestData(test_data, test_labels);
	int rows = data.getRowCount();
	int cols = data.getColCount();

	NeuralNet nn(input_count,
				 output_count,
				 layer_size,
				 p_bias,
				 p_momentum,
				 learning_rate,
				 max_epoch_count,
				 err_thre,
				 batch_size,
				 activation_type,
				 weight_change_para,
				 activation_para);

	nn.stocGradDescTrain(data, 1);

	return 1;
}

bool loadParameters(std::string& training_data,
					std::string& training_labels,
					std::string& test_data,
					std::string& test_labels,
					int& input_count,
					int& output_count,
					std::vector<int>& layer_size,
					double& p_bias,
					double& p_momentum,
					double& learning_rate,
					int& max_epoch_count,
					double& err_thre,
					int& batch_size,
					std::string& activation_type,
					std::vector<std::vector<int>>& weight_change_para,
					std::vector<std::vector<int>>& activation_para) {

	std::string inputPath = "../data/INPUTPARAMETERS.txt";
	std::ifstream infile(inputPath);
	std::string line;

	while (std::getline(infile, line)) {
		if (line == "training_data") {
			std::getline(infile, line);
			training_data = line;
		} else if (line == "training_labels") {
			std::getline(infile, line);
			training_labels = line;
		} else if (line == "test_data") {
			std::getline(infile, line);
			test_data = line;
		} else if (line == "test_labels") {
			std::getline(infile, line);
			test_labels = line;
		} else if (line == "input_count") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> input_count;
		} else if (line == "output_count") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> output_count;
		} else if (line == "layer_size") {
			int tmp;
			std::getline(infile, line);
			std::istringstream iss(line);
			while (iss) {
				iss >> tmp;
				layer_size.push_back(tmp);
			}
		} else if (line == "bias") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> p_bias;
		} else if (line == "momentum") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> p_momentum;
		} else if (line == "learning_rate") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> learning_rate;
		} else if (line == "max_epoch_count") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> max_epoch_count;
		} else if (line == "error_threshold") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> err_thre;
		} else if (line == "batch_size") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> batch_size;
		} else if (line == "activation_type") {
			std::getline(infile, line);
			std::istringstream iss(line);
			iss >> activation_type;
		} else if (line == "weight_changes") {
			int tmp1, tmp2;
			while (std::getline(infile, line)) {
				if (line == "#") break;
				if (line == "") continue;
				std::istringstream iss(line);
				iss >> tmp1 >> tmp2;
				std::vector<int> tmpVec = {tmp1, tmp2};
				weight_change_para.push_back(tmpVec);
			}
		} else if (line == "activation_changes") {
			int tmp1, tmp2, tmp3;
			while (std::getline(infile, line)) {
				if (line == "#") break;
				if (line == "") continue;
				std::istringstream iss(line);
				iss >> tmp1 >> tmp2 >> tmp3;
				std::vector<int> tmpVec = {tmp1, tmp2, tmp3};
				activation_para.push_back(tmpVec);
			}
		}

		}
	// cout << layer_size.size() << endl;
	return true;
}
