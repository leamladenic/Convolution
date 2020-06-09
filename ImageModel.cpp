#include "ImageModel.h"
#include <iostream>
#include <cassert>
#include <stdlib.h>

Image *Image_new(int width, int height, int channels)
{
    float *data = (float *)malloc(sizeof(float) * width * height * channels);
    Image *img = (Image *)malloc(sizeof(Image));

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->pitch = width * channels;
    img->data = data;

    return img;
}

void Image_delete(Image *img)
{
    if (img != NULL)
    {
        if (img->data != NULL)
        {
            free(img->data);
        }
        free(img);
    }
}