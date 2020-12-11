from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot

# 판별 모델 구성
def discriminator(image_shape):
	# kernel_initializaer로 가중치 초기화 => RandomNormal
	init = RandomNormal(stddev=0.02)
	# 스케치 이미지 인풋 모델
	in_src_image = Input(shape=image_shape)
	# 사진 이미지 인풋 모델
	in_target_image = Input(shape=image_shape)
	# 스케치와 사진 인풋 병합
	merged = Concatenate()([in_src_image, in_target_image])
	
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	
	# 모델 정의
	model = Model([in_src_image, in_target_image], patch_out)
	# 컴파일
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

	# 컴파일 한 모델 반환 (훈련x)
	return model
 
# 인코더 구성 : U-Net방식. 크기를 줄임.
def encoder(layer_in, n_filters, batchnorm=True):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# 이전 인코더 노드를 받아 오는 레이어
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# 배치노멀라이제이션이 트루면 적용
	if batchnorm:
		g = BatchNormalization()(g, training=True)

	g = LeakyReLU(alpha=0.2)(g)

	# 레이어 반환
	return g
 
# 디코더 : 크기를 원래대로
def decoder(layer_in, skip_in, n_filters, dropout=True):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# 이전 디코더를 받아 크기를 올리는 레이어
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# 드롭아웃이 트루면 드롭아웃
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# 디코더 레이어와 인코더 레이어를 가까운 순으로 병합
	g = Concatenate()([g, skip_in])

	g = Activation('relu')(g)
	# 레이어 반환
	return g
 
# 생성 모델 구성
def generator(image_shape=(256,256,3)):

	init = RandomNormal(stddev=0.02)

	in_image = Input(shape=image_shape)

	e1 = encoder(in_image, 64, batchnorm=False)
	e2 = encoder(e1, 128)
	e3 = encoder(e2, 256)
	e4 = encoder(e3, 512)
	e5 = encoder(e4, 512)
	e6 = encoder(e5, 512)
	e7 = encoder(e6, 512)

	# 인코딩 결과인 최종 레이어를 새로운 레이어에 연결 
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
	#디코딩 시작
	d1 = decoder(b, e7, 512)
	d2 = decoder(d1, e6, 512)
	d3 = decoder(d2, e5, 512)
	d4 = decoder(d3, e4, 512, dropout=False)
	d5 = decoder(d4, e3, 256, dropout=False)
	d6 = decoder(d5, e2, 128, dropout=False)
	d7 = decoder(d6, e1, 64, dropout=False)
	
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)

	model = Model(in_image, out_image)
	return model
 
# 판별자, 생성자 합치는 모델
def gan(g_model, d_model, image_shape):
	# 판별자 모델 동결
	d_model.trainable = False
	# 스케치 이미지 인풋 모델 생성
	in_src = Input(shape=image_shape)
	# 스케치 인풋 모델을 생성자 모델 아웃풋과 연결
	gen_out = g_model(in_src)
	# 스케치 인풋 모델과 생성자 모델 아웃풋을 판별자 모델 아웃풋과 연결
	dis_out = d_model([in_src, gen_out])
	# 인풋에 스케치, 아웃풋에 생성자, 판별자 모델로 분류해서 모델 선언
	model = Model(in_src, [dis_out, gen_out])
	# 컴파일
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	#모델로 반환
	return model
 
# 훈련 시킬 사진 이미지 불러오기
def load_real_samples(filename):

	data = load(filename)
	# 스케치(X1)와 사진(X2) 분류
	X1, X2 = data['arr_0'], data['arr_1']
	# 스케일링 : [0,255] 을 [-1,1]로
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# 진짜 이미지 샘플에서 랜덤으로 배치사이즈 선택. data와 target으로 리턴
def generate_real_samples(dataset, n_samples, patch_shape):

	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y
 
# 가짜 샘플 이미지 배치 생성, data와 target 리턴
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
 
# 샘플 생성하고 모델과 샘플 저장
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# pix2pix 모델 학습시키기
def train(d_model, g_model, gan_model, dataset, n_epochs=150, n_batch=16):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)
 
######### 1. 데이터 : 스케치-사진 페어 데이터셋 로드
dataset = load_real_samples('teapot_strawberry.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

image_shape = dataset[0].shape[1:]

######### 2. 모델 구성

# 모델 1. 판별자
d_model = discriminator(image_shape)
# 모델 2. 생성자
g_model = generator(image_shape)
# 모델 3. 판별자+생성자를 합치는 모델
gan_model = gan(g_model, d_model, image_shape)

######### 3. 모델 전체 훈련
train(d_model, g_model, gan_model, dataset)