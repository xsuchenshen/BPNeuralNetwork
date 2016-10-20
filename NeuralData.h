#ifndef __NEURALDATA_H__
#define __NEURALDATA_H__

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

#define TOTAL_CLASS_COUNT 10

class NeuralData {
private:
	int32_t magicNumberImage;
	int32_t magicNumberLabel;
	int32_t imageCount;
	int32_t rowCount;
	int32_t colCount;
	int32_t labelCount;

	std::vector<int> lblVec;

public:
	NeuralData();
	~NeuralData();

	std::vector<std::vector<double>> imgMat;
	std::vector<std::vector<double>> tgtMat;

	std::vector<std::vector<double>> imgMatTest;
	std::vector<int> tgtVecTest;

	void loadTrainingData(std::string dataName, std::string labelName);
	void loadTestData(std::string dataName, std::string labelName);
	int getImageCount() { return this->imageCount; }
	int getRowCount() { return this->rowCount; }
	int getColCount() { return this->colCount; }

protected:
	// high endian to low endian
	int32_t flipBytes(int32_t val);
	bool loadTrainingImageMNIST(std::string fileName);
	bool loadTrainingLabelMNIST(std::string fileName);
	bool loadTestImageMNIST(std::string fileName);
	bool loadTestLabelMNIST(std::string fileName);
	bool shuffleData();
};

#endif
