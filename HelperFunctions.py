#HelperFunctions

from PIL import Image

def printMNISTVector(imageVector):
	img = Image.new('L', (28, 28))
	pixels = img.load()

	vectori = 0
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			pixels[j, i] = int(imageVector[vectori]*255)
			vectori += 1

	img.show()
