#include "visual_odometry/io.h"
#include <algorithm>

#ifdef __MINGW32__
#include <sys/stat.h>
#endif

// functions to read images from the folder data ------------------------------------------
std::string readWriteFunctions::toLowerCase(const std::string& in) {
    std::string t;
    for (std::string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

void readWriteFunctions::getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions)
{
    printf("Opening directory %s\n", dirName.c_str());
#ifdef __MINGW32__
    struct stat s;
#endif
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
#ifdef __MINGW32__
            stat(ep->d_name, &s);
            if (s.st_mode & S_IFDIR) {
                continue;
            }
#else
            if (ep->d_type & DT_DIR) {
                continue;
            }
#endif
            extensionLocation = std::string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            std::string tempExt = toLowerCase(std::string(ep->d_name).substr(extensionLocation + 1));
            if (std::find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                //                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((std::string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}
// --------------------------------------------------------------------------------------
