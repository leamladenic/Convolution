#ifndef IMAGE_H_
#define IMAGE_H_

typedef struct
{
    int width;
    int height;
    int channels;
    int pitch;
    float *data;
} Image;

#define IMAGE_CHANNELS 3

Image *Image_new(int width, int height, int channels);
void Image_delete(Image *img);

#endif