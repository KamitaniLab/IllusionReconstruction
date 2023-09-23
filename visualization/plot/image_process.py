import cv2
import numpy as np

def normalise_img(img):
    '''Normalize the image.
    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int
    '''
    img = img - img.min()
    if img.max() > 0:
        img = img * (255.0/img.max())
    img = np.uint8(img)
    return img

def img_process(img_path, img_size=None, gray=None, redness=None):
    """
    Read image, resize and turn into grayscale
    Inputs:
        img_path 
        img_size - (width, height)
    """
    
    # read image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # resize image
    if not img_size is None:

        if type(img_size) is tuple:
            dim = img_size
        else:
            dim = (img_size, img_size)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
    # grayscale
    if not gray is None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if redness == 'Redness1':
        denom =  img[:,:,2]+(img[:,:,0] + img[:,:,1])/2
        img = (img[:,:,2]-(img[:,:,0] + img[:,:,1])/2)/denom
        img[denom==0] = 0
        # rescale to [0,255]
        img = normalise_img(img)
        
    elif redness == 'Redness2':
        img = np.minimum(255, np.maximum(0, img[:,:,2]-img[:,:,0]-img[:,:,1]))
        
    elif redness == 'Redness':      
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
         
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 0, 0])
        upper1 = np.array([10, 255, 255])
         
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 0, 0])
        upper2 = np.array([179,255,255])
         
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)
         
        full_mask = lower_mask + upper_mask;
         
        img = cv2.bitwise_and(image[:,:,1], image[:,:,1], mask=full_mask)

    return img

def gammaCorrection(src, gamma=2.2):
    table = [(i/255)**(1/gamma)*255 for i in range(256)]
    table = np.array(table, np.uint8)
    
    return cv2.LUT(src, table)


def cvtHSVtoRGB(hue_list, saturation, value):
    
    rgb_list = dict.fromkeys(hue_list)
    for hue in hue_list:
        rgb_list[hue] = cv2.cvtColor(np.uint8([[[hue, saturation,value]]]),
                                     cv2.COLOR_HSV2RGB)
    
    return rgb_list
