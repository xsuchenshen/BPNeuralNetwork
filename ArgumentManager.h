#include <map>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;

class ArgumentManager
{
    private:
        map<string, string> argumentMap;
    public:
        ArgumentManager(){ }
        ArgumentManager(int _argc, char *argv[], char delimiter=';');
        ArgumentManager(string rawArguments, char delimiter=';');
        void parse(int argc, char *argv[], char delimiter=';');
        void parse(string rawArguments, char delimiter=';');
        string get(string argumentName);
        string toString();
        friend ostream& operator << (ostream &out, ArgumentManager &am);
};

void ArgumentManager::parse(string rawArguments, char delimiter)
{
    stringstream currentArgumentName;
    stringstream currentArgumentValue;
    bool argumentNameFinished = false;
    
    for(unsigned int i=0; i<=rawArguments.length(); ++i)
    {
        if(i == rawArguments.length() || rawArguments[i] == delimiter)
        {
            if(currentArgumentName.str() != "")
            {
                argumentMap[currentArgumentName.str()] = currentArgumentValue.str();
            }
            // reset
            currentArgumentName.str("");
            currentArgumentValue.str("");
            argumentNameFinished = false;
        }
        else if(rawArguments[i] == '=')
        {
            argumentNameFinished = true;
        }
        else
        {
            if(argumentNameFinished)
            {
                currentArgumentValue << rawArguments[i];
            }
            else
            {
                // ignore any spaces in argument names. 
                if(rawArguments[i] == ' ')
                    continue;
                currentArgumentName << rawArguments[i];
            }
        }
    }
}

void ArgumentManager::parse(int argc, char *argv[], char delimiter)
{
    if(argc > 1)
    {
        for(int i=1; i<argc; i++)
        {
            parse(argv[i], delimiter);
        }
    }
}

ArgumentManager::ArgumentManager(int argc, char *argv[], char delimiter)
{
    parse(argc, argv, delimiter);
}

ArgumentManager::ArgumentManager(string rawArguments, char delimiter)
{
    parse(rawArguments, delimiter);
}

string ArgumentManager::get(string argumentName)
{
    map<string, string>::iterator iter = argumentMap.find(argumentName);

    //If the argument is not found, return a blank string.
    if(iter == argumentMap.end())
    {
        return "";
    }
    else
    {
        return iter->second;
    }
}

string ArgumentManager::toString()
{
    stringstream ss;
    for(map<string, string>::iterator iter = argumentMap.begin(); iter != argumentMap.end(); ++iter)
    {
        ss << "Argument name: " << iter->first << endl;
        ss << "Argument value: " << iter->second << endl;
    }
    return ss.str();
}

ostream& operator << (ostream &out, ArgumentManager &am)
{
    out << am.toString();
    return out;
}
