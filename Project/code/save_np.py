import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LeakyReLU, Dropout, Input, Concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import savez_compressed


#이미지 이름 저장
camel_sketch_idx = next(os.walk("./data/sketch/camel/"))[2]
camel_photo_idx = next(os.walk("./data/photo/camel/"))[2]
cat_sketch_idx = next(os.walk("./data/sketch/cat/"))[2]
cat_photo_idx = next(os.walk("./data/photo/cat/"))[2]
cup_sketch_idx = next(os.walk("./data/sketch/cup/"))[2]
cup_photo_idx = next(os.walk("./data/photo/cup/"))[2]
dog_sketch_idx = next(os.walk("./data/sketch/dog/"))[2]
dog_photo_idx = next(os.walk("./data/photo/dog/"))[2]
door_sketch_idx = next(os.walk("./data/sketch/door/"))[2]
door_photo_idx = next(os.walk("./data/photo/door/"))[2]
flower_sketch_idx = next(os.walk("./data/sketch/flower/"))[2]
flower_photo_idx = next(os.walk("./data/photo/flower/"))[2]
horse_sketch_idx = next(os.walk("./data/sketch/horse/"))[2]
horse_photo_idx = next(os.walk("./data/photo/horse/"))[2]
jellyfish_sketch_idx = next(os.walk("./data/sketch/jellyfish/"))[2]
jellyfish_photo_idx = next(os.walk("./data/photo/jellyfish/"))[2]
piano_sketch_idx = next(os.walk("./data/sketch/piano/"))[2]
piano_photo_idx = next(os.walk("./data/photo/piano/"))[2]
rabbit_sketch_idx = next(os.walk("./data/sketch/rabbit/"))[2]
rabbit_photo_idx = next(os.walk("./data/photo/rabbit/"))[2]



camel_sketch = np.zeros((len(camel_sketch_idx),256,256,3),dtype=np.uint8) 
camel_photo = np.zeros((len(camel_sketch_idx),256,256,3),dtype=np.uint8)
cat_sketch = np.zeros((len(cat_sketch_idx),256,256,3),dtype=np.uint8) 
cat_photo = np.zeros((len(cat_sketch_idx),256,256,3),dtype=np.uint8)
cup_sketch = np.zeros((len(cup_sketch_idx),256,256,3),dtype=np.uint8) 
cup_photo = np.zeros((len(cup_sketch_idx),256,256,3),dtype=np.uint8)
dog_sketch = np.zeros((len(dog_sketch_idx),256,256,3),dtype=np.uint8) 
dog_photo = np.zeros((len(dog_sketch_idx),256,256,3),dtype=np.uint8)
door_sketch = np.zeros((len(door_sketch_idx),256,256,3),dtype=np.uint8) 
door_photo = np.zeros((len(door_sketch_idx),256,256,3),dtype=np.uint8)
flower_sketch = np.zeros((len(flower_sketch_idx),256,256,3),dtype=np.uint8) 
flower_photo = np.zeros((len(flower_sketch_idx),256,256,3),dtype=np.uint8)
horse_sketch = np.zeros((len(horse_sketch_idx),256,256,3),dtype=np.uint8) 
horse_photo = np.zeros((len(horse_sketch_idx),256,256,3),dtype=np.uint8)
jellyfish_sketch = np.zeros((len(jellyfish_sketch_idx),256,256,3),dtype=np.uint8) 
jellyfish_photo = np.zeros((len(jellyfish_sketch_idx),256,256,3),dtype=np.uint8)
piano_sketch = np.zeros((len(piano_sketch_idx),256,256,3),dtype=np.uint8) 
piano_photo = np.zeros((len(piano_sketch_idx),256,256,3),dtype=np.uint8)
rabbit_sketch = np.zeros((len(rabbit_sketch_idx),256,256,3),dtype=np.uint8) 
rabbit_photo = np.zeros((len(rabbit_sketch_idx),256,256,3),dtype=np.uint8)

from tqdm import tqdm_notebook
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize #사이즈 조절

def sketch_image(sketch_idx, name, sketch) :
    for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
        print("i : ",i)
        print("idx : ",idx)
        img = load_img("./data/sketch/"+name+"/"+idx)
        img = img_to_array(img)
        sketch[i] = img
    return sketch

def photo_image(sketch_idx, name, photo) :
    for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
        print("i : ",i)
        print("idx : ",idx)
        idx = idx.split("-")[0]
        img = load_img("./data/photo/"+name+"/"+ idx+".jpg")
        img = img_to_array(img)
        photo[i] = img
    return photo



#sketch: X, photo: Y
np.save('./npySAVE/sketch_camel.npy', arr=sketch_image(camel_sketch_idx, 'camel', camel_sketch))
np.save('./npySAVE/photo_camel.npy', arr=photo_image(camel_sketch_idx, 'camel', camel_photo))
np.save('./npySAVE/sketch_cat.npy', arr=sketch_image(cat_sketch_idx, 'cat', cat_sketch))
np.save('./npySAVE/photo_cat.npy', arr=photo_image(cat_sketch_idx, 'cat', cat_photo))
np.save('./npySAVE/sketch_cup.npy', arr=sketch_image(cup_sketch_idx, 'cup', cup_sketch))
np.save('./npySAVE/photo_cup.npy', arr=photo_image(cup_sketch_idx, 'cup', cup_photo))
np.save('./npySAVE/sketch_dog.npy', arr=sketch_image(dog_sketch_idx, 'dog', dog_sketch))
np.save('./npySAVE/photo_dog.npy', arr=photo_image(dog_sketch_idx, 'dog', dog_photo))
np.save('./npySAVE/sketch_door.npy', arr=sketch_image(door_sketch_idx, 'door', door_sketch))
np.save('./npySAVE/photo_door.npy', arr=photo_image(door_sketch_idx, 'door', door_photo))
np.save('./npySAVE/sketch_flower.npy', arr=sketch_image(flower_sketch_idx, 'flower', flower_sketch))
np.save('./npySAVE/photo_flower.npy', arr=photo_image(flower_sketch_idx, 'flower', flower_photo))
np.save('./npySAVE/sketch_horse.npy', arr=sketch_image(horse_sketch_idx, 'horse', horse_sketch))
np.save('./npySAVE/photo_horse.npy', arr=photo_image(horse_sketch_idx, 'horse', horse_photo))
np.save('./npySAVE/sketch_jellyfish.npy', arr=sketch_image(jellyfish_sketch_idx, 'jellyfish', jellyfish_sketch))
np.save('./npySAVE/photo_jellyfish.npy', arr=photo_image(jellyfish_sketch_idx, 'jellyfish', jellyfish_photo))
np.save('./npySAVE/sketch_piano.npy', arr=sketch_image(piano_sketch_idx, 'piano', piano_sketch))
np.save('./npySAVE/photo_piano.npy', arr=photo_image(piano_sketch_idx, 'piano', piano_photo))
np.save('./npySAVE/sketch_rabbit.npy', arr=sketch_image(rabbit_sketch_idx, 'rabbit', rabbit_sketch))
np.save('./npySAVE/photo_rabbit.npy', arr=photo_image(rabbit_sketch_idx, 'rabbit', rabbit_photo))

filename = 'sketch_to_photo.npz'
np.savez_compressed(filename, sketch_image(camel_sketch_idx, 'camel', camel_sketch), photo_image(camel_sketch_idx, 'camel', camel_photo),
                            sketch_image(cat_sketch_idx, 'cat', cat_sketch), photo_image(cat_sketch_idx, 'cat', cat_photo),
                            sketch_image(cup_sketch_idx, 'cup', cup_sketch), photo_image(cup_sketch_idx, 'cup', cup_photo),
                            sketch_image(dog_sketch_idx, 'dog', dog_sketch), photo_image(dog_sketch_idx, 'dog', dog_photo),
                            sketch_image(door_sketch_idx, 'door', door_sketch), photo_image(door_sketch_idx, 'door', door_photo),
                            sketch_image(flower_sketch_idx, 'flower', flower_sketch), photo_image(flower_sketch_idx, 'flower', flower_photo),
                            sketch_image(horse_sketch_idx, 'horse', horse_sketch),photo_image(horse_sketch_idx, 'horse', horse_photo),
                            sketch_image(jellyfish_sketch_idx, 'jellyfish', jellyfish_sketch), photo_image(jellyfish_sketch_idx, 'jellyfish', jellyfish_photo),
                            sketch_image(piano_sketch_idx, 'piano', piano_sketch), photo_image(piano_sketch_idx, 'piano', piano_photo),
                            sketch_image(rabbit_sketch_idx, 'rabbit', rabbit_sketch), photo_image(rabbit_sketch_idx, 'rabbit', rabbit_photo))



