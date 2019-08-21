#include <dirent.h>
#include <string>
#include <vector>

class readWriteFunctions
{
public:
    static std::string toLowerCase(const std::string& in);
    static void getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions);
};
