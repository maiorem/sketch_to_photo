from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import sys


# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)

	# convert to numpy array
	pixels = img_to_array(pixels)
	print(pixels)
	print(pixels.shape)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	print(pixels)
	print(pixels.shape)
	return pixels

# load source image
src_image = load_image('strawberry.png')
print('Loaded', src_image.shape)

ef=load_model('bear_strawberry_softmax.h5')
label=ef.predict(src_image)

teapot=format(label[0][0] * 100, ".1f")
strawberry=format(label[0][1] * 100, ".1f")
print('딸기일 확률', teapot, '%', '테디베어일 확률', strawberry, '%')

# load model
model = load_model('model_087200.h5')
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
# plot the image
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()

