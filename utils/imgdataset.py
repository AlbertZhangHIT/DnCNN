import os
import os.path
import numpy as np
import h5py
import torch
import cv2
from PIL import Image
import torch.utils.data as udata
from . import functional  as F

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def default_loader(path):
# 	img = cv2.imread(path)
# 	c = img.shape[-1]
# 	if c > 1:
# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
# 	Img = img[:, :, 0:1]
# 	return Img

class pil_Compose(object):
    def __init__(self, operations):
        self.operations = operations

    def __call__(self, img):
        for op in self.operations:
            img = op(img)

        return img
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.operations:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

def pil_loader(path, mode='YCbCr'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)

class pil_rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return img.rotate(self.angle)

class pil_transpose(object):
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        valid = [0, 1, 2, 3, 4]
        if self.mode in valid:
            if self.mode==0:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.mode==1:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            if self.mode==2:
                return img.transpose(Image.ROTATE_90)
            if self.mode==3:
                return img.transpose(Image.ROTATE_180)    
            if self.mode==4:
                return img.transpose(Image.ROTATE_270)
        else:
            print("The mode exceeds 4 will do nothing.")

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageDatasetFromFolder(udata.Dataset):
	def __init__(self, dataPath, loader=default_loader, transform=None):
		super(ImageDatasetFromFolder, self).__init__()
		self.path = dataPath
		self.transform = transform
		self.samples = [os.path.join(self.path, x) for x in os.listdir(self.path) if is_image_file(x)]
		self.loader = loader

	def __getitem__(self, index):
		imgFile = self.samples[index]
		sample = self.loader(imgFile)
		if self.transform is not None:
			sample = self.transform(sample)
		return sample

	def __len__(self):
		return len(self.samples) 