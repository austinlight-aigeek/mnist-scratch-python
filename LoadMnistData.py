from struct import unpack
import gzip
from numpy import uint8, zeros, float32

def LoadMnistData(imagefile, labelfile):
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')
    
    images.read(4)
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]
    
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]
    
    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')
        
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
            
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel

        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    return (x, y)
    