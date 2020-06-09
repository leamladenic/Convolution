#ifndef PPM_H_
#define PPM_H_

#include "ImageModel.h"

Image *importPPM(const char *filename);
bool exportPPM(const char *filename, Image *img);

#endif