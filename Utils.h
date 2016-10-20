#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <stack>
#include <sstream>

using namespace std;

class Utils {
public:
	static bool explode(std::string s, char delim, std::vector<std::string>& res);

	static bool lTrim(std::string& s);

	static bool rTrim(std::string& s);

	static bool trim(std::string& s);

	static bool matchInnerPair(std::string s, std::string ls, std::string rs, std::string& res);

	static bool getLeftString(std::string s, std::string delim, std::string& res);

	static bool getRightString(std::string s, std::string delim, std::string& res);

	static bool isIntNumber(std::string s);
};

#endif