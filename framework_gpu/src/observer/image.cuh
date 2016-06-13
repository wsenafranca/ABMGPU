#ifndef IMAGE_CUH
#define IMAGE_CUH

template<class T>
int saveBMP(T *map, unsigned int width, unsigned int height, const char *filename) {
    unsigned int filesize = 54 + 3*width*height;
    
    unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
    
    for(unsigned int i = 0; i < height*width; i++) {
        img[i*3+2] = 255;
        img[i*3+1] = 255;
        img[i*3+0] = 255;
    }
    
    for(unsigned int i = 0; i < height*width; i++) {
        if(map[i] > 0) {
            //printf("%d\n", cells[i].quantity);
            int b = 1;//cells[i].quantity*BOID_SIZE;
            for(int j = -b/2; j <= b/2; j++) {
                for(int k = -b/2; k <= b/2; k++) {
                    int x = i%width;
                    int y = i/width;
                    int idx = (y+j)*width+(x+k);
                    if(idx >= 0 && idx < width*height) {
                        img[idx*3+2] = 0;
                        img[idx*3+1] = 0;
                        img[idx*3+0] = 0;
                    }
                }
            }
        }
    }
    
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};
    
    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(width);
    bmpinfoheader[ 5] = (unsigned char)(width>> 8);
    bmpinfoheader[ 6] = (unsigned char)(width>>16);
    bmpinfoheader[ 7] = (unsigned char)(width>>24);
    bmpinfoheader[ 8] = (unsigned char)(height);
    bmpinfoheader[ 9] = (unsigned char)(height>> 8);
    bmpinfoheader[10] = (unsigned char)(height>>16);
    bmpinfoheader[11] = (unsigned char)(height>>24);
    
    FILE *f = fopen(filename,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(unsigned i=0; i<height; i++) {
        fwrite(img+(width*(height-i-1)*3),3,width,f);
        fwrite(bmppad,1,(4-(width*3)%4)%4,f);
    }
    fclose(f);
    
    free(img);
    return 0;
}

#endif

