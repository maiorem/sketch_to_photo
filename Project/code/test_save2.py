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
camel_photo1_idx = next(os.walk("./data2/photo/camel/"))[2]
door_photo1_idx = next(os.walk("./data2/photo/door/"))[2]
camel_sketch1_idx = next(os.walk("./data2/sketch/0/camel/"))[2]
door_sketch1_idx = next(os.walk("./data2/sketch/0/door/"))[2]

camel_photo2_idx = next(os.walk("./data2/photo/camel/"))[2]
door_photo2_idx = next(os.walk("./data2/photo/door/"))[2]
camel_sketch2_idx = next(os.walk("./data2/sketch/1/camel/"))[2]
door_sketch2_idx = next(os.walk("./data2/sketch/1/door/"))[2]

camel_photo3_idx = next(os.walk("./data2/photo/camel/"))[2]
door_photo3_idx = next(os.walk("./data2/photo/door/"))[2]
camel_sketch3_idx = next(os.walk("./data2/sketch/2/camel/"))[2]
door_sketch3_idx = next(os.walk("./data2/sketch/3/door/"))[2]



camel_sketch1 = np.zeros((len(camel_sketch1_idx),256,256,3),dtype=np.uint8) 
camel_photo1 = np.zeros((len(camel_sketch1_idx),256,256,3),dtype=np.uint8)
door_sketch1 = np.zeros((len(door_sketch1_idx),256,256,3),dtype=np.uint8) 
door_photo1 = np.zeros((len(door_sketch1_idx),256,256,3),dtype=np.uint8)

camel_sketch2 = np.zeros((len(camel_sketch2_idx),256,256,3),dtype=np.uint8) 
camel_photo2 = np.zeros((len(camel_sketch2_idx),256,256,3),dtype=np.uint8)
door_sketch2 = np.zeros((len(door_sketch2_idx),256,256,3),dtype=np.uint8) 
door_photo2 = np.zeros((len(door_sketch2_idx),256,256,3),dtype=np.uint8)

camel_sketch3 = np.zeros((len(camel_sketch3_idx),256,256,3),dtype=np.uint8) 
camel_photo3 = np.zeros((len(camel_sketch3_idx),256,256,3),dtype=np.uint8)
door_sketch3 = np.zeros((len(door_sketch3_idx),256,256,3),dtype=np.uint8) 
door_photo3 = np.zeros((len(door_sketch3_idx),256,256,3),dtype=np.uint8)



from tqdm import tqdm_notebook
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize #사이즈 조절

def sketch_image(sketch_idx, num, name, sketch) :
    for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
        print("i : ",i)
        print("idx : ",idx)
        img = load_img("./data2/sketch/"+num+"/"+name+"/"+idx)
        img = img_to_array(img)
        sketch[i] = img
    return sketch

def photo_image(sketch_idx, name, photo) :
    for i, idx in tqdm_notebook(enumerate(sketch_idx),total=len(sketch_idx)):
        print("i : ",i)
        print("idx : ",idx)
        idx = idx.split("-")[0]
        img = load_img("./data2/photo/"+name+"/"+ idx+".jpg")
        img = img_to_array(img)
        photo[i] = img
    return photo


camel_sketch1=sketch_image(camel_sketch1_idx, '0', 'camel', camel_sketch1)
camel_photo1=photo_image(camel_sketch1_idx, 'camel', camel_photo1)
door_sketch1=sketch_image(door_sketch1_idx, '0', 'door', door_sketch1)
door_photo1=photo_image(door_sketch1_idx, 'door', door_photo1)

camel_sketch2=sketch_image(camel_sketch2_idx, '1', 'camel', camel_sketch2)
camel_photo2=photo_image(camel_sketch2_idx, 'camel', camel_photo2)
door_sketch2=sketch_image(door_sketch2_idx, '1', 'door', door_sketch2)
door_photo2=photo_image(door_sketch2_idx, 'door', door_photo2)

camel_sketch3=sketch_image(camel_sketch3_idx, '2', 'camel', camel_sketch3)
camel_photo3=photo_image(camel_sketch3_idx, 'camel', camel_photo3)
door_sketch3=sketch_image(door_sketch3_idx, '2', 'door', door_sketch3)
door_photo3=photo_image(door_sketch3_idx, 'door', door_photo3)

#sketch: X, photo: Y
# np.save('./npySAVE/sketch_camel1.npy', arr=camel_sketch1)
# np.save('./npySAVE/photo_camel1.npy', arr=camel_photo1)
# np.save('./npySAVE/sketch_door1.npy', arr=door_sketch1)
# np.save('./npySAVE/photo_door1.npy', arr=door_photo1)

# np.save('./npySAVE/sketch_camel2.npy', arr=camel_sketch2)
# np.save('./npySAVE/photo_camel2.npy', arr=camel_photo2)
# np.save('./npySAVE/sketch_door2.npy', arr=door_sketch2)
# np.save('./npySAVE/photo_door2.npy', arr=door_photo2)

# np.save('./npySAVE/sketch_camel3.npy', arr=camel_sketch3)
# np.save('./npySAVE/photo_camel3.npy', arr=camel_photo3)
# np.save('./npySAVE/sketch_door3.npy', arr=door_sketch3)
# np.save('./npySAVE/photo_door3.npy', arr=door_photo3)


filename = 'sketch_to_photo.npz'
np.savez_compressed(filename, [camel_sketch1, door_sketch1, camel_sketch2, door_sketch2, camel_sketch3, door_sketch3], [camel_photo1, door_photo1, camel_photo2, door_photo2, camel_photo3, door_photo3])



