#include "NeuralData.h"

NeuralData::NeuralData() {

}

NeuralData::~NeuralData() {

}

void NeuralData::loadTrainingData(std::string dataName, std::string labelName) {
	this->loadTrainingImageMNIST(dataName);
	this->loadTrainingLabelMNIST(labelName);
	this->shuffleData();

	//this->imgMat.resize(60000);
	//this->tgtMat.resize(60000);
}

void NeuralData::loadTestData(std::string dataName, std::string labelName) {
	this->loadTestImageMNIST(dataName);
	this->loadTestLabelMNIST(labelName);
}

bool NeuralData::shuffleData() {
	std::vector<int> vec(this->imageCount, 0);
	std::vector<std::vector<double>> imgTmp(this->imgMat);
	std::vector<std::vector<double>> tgtTmp(this->tgtMat);

	for (int i = 0; i < vec.size(); i++) {
		vec[i] = i;
	}
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));

	for (int i = 0; i < this->imageCount; i++) {
		imgMat[i] = imgTmp[vec[i]];
		tgtMat[i] = tgtTmp[vec[i]];
	}

	return true;
}

bool NeuralData::loadTrainingImageMNIST(std::string fileName) {
	std::ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		file.read((char*)&(this->magicNumberImage), sizeof(this->magicNumberImage));
		file.read((char*)&(this->imageCount), sizeof(this->imageCount));
		file.read((char*)&(this->rowCount), sizeof(this->rowCount));
		file.read((char*)&(this->colCount), sizeof(this->colCount));
		this->magicNumberImage = this->flipBytes(this->magicNumberImage);
		this->imageCount = this->flipBytes(this->imageCount);
		this->rowCount = this->flipBytes(this->rowCount);
		this->colCount = this->flipBytes(this->colCount);

		for (int i = 0; i < (int)(this->imageCount); i++) {
			std::vector<double> tmpVec;
			for (int j = 0; j < (int)(this->rowCount); j++) {
				for (int k = 0; k < (int)(this->colCount); k++) {
					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));
					tmpVec.push_back(((double)tmp / 255.0));
				}
			}
			this->imgMat.push_back(tmpVec);
		}
	} else {
		return false;
	}

	return true;
}

bool NeuralData::loadTestImageMNIST(std::string fileName) {
	std::ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		file.read((char*)&(this->magicNumberImage), sizeof(this->magicNumberImage));
		file.read((char*)&(this->imageCount), sizeof(this->imageCount));
		file.read((char*)&(this->rowCount), sizeof(this->rowCount));
		file.read((char*)&(this->colCount), sizeof(this->colCount));
		this->magicNumberImage = this->flipBytes(this->magicNumberImage);
		this->imageCount = this->flipBytes(this->imageCount);
		this->rowCount = this->flipBytes(this->rowCount);
		this->colCount = this->flipBytes(this->colCount);

		for (int i = 0; i < (int)(this->imageCount); i++) {
			std::vector<double> tmpVec;
			for (int j = 0; j < (int)(this->rowCount); j++) {
				for (int k = 0; k < (int)(this->colCount); k++) {
					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));
					tmpVec.push_back(((double)tmp / 255.0));
				}
			}
			this->imgMatTest.push_back(tmpVec);
		}
	} else {
		std::cout << "ERROR LOADING FILE\n";
		return false;
	}

	return true;
}

bool NeuralData::loadTrainingLabelMNIST(std::string fileName) {
	std::ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		file.read((char*)&(this->magicNumberLabel), sizeof(this->magicNumberLabel));
		file.read((char*)&(this->labelCount), sizeof(this->labelCount));
		this->magicNumberLabel = this->flipBytes(this->magicNumberLabel);
		this->labelCount = this->flipBytes(this->labelCount);

		for (int i = 0; i < (int)(this->labelCount); i++) {
			std::vector<double> tmpVec(TOTAL_CLASS_COUNT, 0.1);
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			this->lblVec.push_back((int)tmp);
			tmpVec[(int)tmp] = 0.9;
			this->tgtMat.push_back(tmpVec);
		}
	} else {
		return false;
	}

	return true;
}

bool NeuralData::loadTestLabelMNIST(std::string fileName) {
	std::ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		file.read((char*)&(this->magicNumberLabel), sizeof(this->magicNumberLabel));
		file.read((char*)&(this->labelCount), sizeof(this->labelCount));
		this->magicNumberLabel = this->flipBytes(this->magicNumberLabel);
		this->labelCount = this->flipBytes(this->labelCount);

		for (int i = 0; i < (int)(this->labelCount); i++) {
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			this->tgtVecTest.push_back((int)tmp);
		}
	} else {
		return false;
	}

	return true;
}

int32_t NeuralData::flipBytes(int32_t val) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = val & 255;
	ch2 = (val >> 8) & 255;
	ch3 = (val >> 16) & 255;
	ch4 = (val >> 24) & 255;

	return ((int32_t)ch1 << 24)
			+ ((int32_t)ch2 << 16)
			+ ((int32_t)ch3 << 8)
			+ ch4;
}
