#include "Utils.h"

bool Utils::explode(std::string s, char delim, std::vector<std::string>& res) {
	res.clear();
	std::istringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		res.push_back(item);
	}
	
	return true;
}

bool Utils::lTrim(std::string& s) {
	if (s == "") {
		return false;
	}
	s = s.substr(s.find_first_not_of(" "));

	return true;
}

bool Utils::rTrim(std::string& s) {
	if (s == "") {
		return false;
	}
	s = s.substr(0, s.find_last_not_of(" ") + 1);

	return true;
}

bool Utils::trim(std::string& s) {
	if (s == "") {
		return false;
	}
	lTrim(s);
	rTrim(s);

	return true;
}

bool Utils::matchInnerPair(std::string s, std::string ls, std::string rs, std::string& res) {
	res.clear();
	std::size_t found1;
	std::size_t found2;
	found1 = s.find(ls);
	found2 = s.find_last_of(rs);
	if (found1 != std::string::npos && found2 != std::string::npos) {
		if (found2 > found1) {
			res = s.substr(found1 + 1, found2 - found1 - 1);
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}

	return true;
}

bool Utils::getLeftString(std::string s, std::string delim, std::string& res) {
	std::size_t found = s.find(delim);
	if (found != std::string::npos) {
		res = s.substr(0, found);
	}
	else {
		return false;
	}

	return true;
}

bool Utils::getRightString(std::string s, std::string delim, std::string& res) {
	std::size_t found = s.find(delim);
	if (found != std::string::npos) {
		res = s.substr(found + delim.size());
	}
	else {
		return false;
	}

	return true;
}

bool Utils::isIntNumber(std::string s) {
	return !s.empty() && s.find_first_not_of("-0123456789") == std::string::npos;
}