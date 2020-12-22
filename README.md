# sketch_to_photo
스케치를 사진으로 변환하는 인공지능 프로젝트

## Team
TeamName : Gan 때문이야       
ProjectName : Sketch to Photo using GAN     
팀장 : 홍세영     
팀원 : 박은지, 신여명, 오수연     

## 사용기술
Language : Python        
Tool : Visual Studio Code      
API : Tensorflow, Keras, Numpy, OpenCv        
Web : Flask, HTML5, Javascript       
Model : GAN_Pix2Pix,VGG16     

## 참조
 * DATA SETS : [스케치와 사진 데이터셋](http://sketchy.eye.gatech.edu/)       
 * 관련논문 :        
     - [Sketch to Image Translation using GANs](https://lisa.fan/Resources/SketchGAN/sketch-image-translation.pdf)      
     - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)       
 * [슬라이드](https://lisa.fan/Resources/SketchGAN/sketchganslides.pdf)        
 * [Tensorflow 공식문서](https://www.tensorflow.org/tutorials/generative/pix2pix)
 * [CoLab 예제](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)         


## ▶ 목표   

### ◎ 스케치 이미지를 사진으로 바꾸는 이미지 생성(GAN) 모델 만들기      
![sketch](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fwnaed%2FbtqQIV2opBd%2F2QsiIbR9Dmz4UdnHmTX8bK%2Fimg.jpg)
<br />
### 단지 새로운 이미지를 생성하는 것이 아니라, 스케치 이미지를 사진 이미지로 변환하는 Image to Image 작업이므로 관련 논문인 Pix2Pix를 참고하여 스케치와 사진 데이터의 쌍을 만들어 모델에 훈련시킨다.         
<br />

## ▶ 데이터 수집과 전처리       

### ◎ [스케치와 사진 데이터셋](http://sketchy.eye.gatech.edu/) 
![dataset](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSDqup%2FbtqQKuYDzlo%2F5bGyjnufrfovOPX1NdE1J0%2Fimg.png)        
 * 총 125개 카테고리와 12500개의 사진, 사진을 보고 그린 스케치들이 존재하는 데이터셋 사용.       
해당 데이터 중 테스트를 위한 네개의 카테고리를 선별.       
<br />

![4_categories](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpfbB0%2FbtqQR63p7G6%2FIp6n4ms9pZW3kKKHKpFhk1%2Fimg.jpg)
<br />

### ◎ 이미지 분할처리 [Image Segmentation](https://github.com/maiorem/sketch_to_photo/blob/main/Project/pix2pix/pix2pix_imagesegmentation.py)
![img seg](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcCoR2m%2FbtqQXBu5omK%2FYXix5FhOgiC2O1RebGyme0%2Fimg.jpg)       
 * 훈련 시킬 사진 이미지에 배경이 강렬할 수록 우리가 원하는 객체를 명확히 인식하여 훈련시키기가 힘들다. 때문에 객체를 주변 사물로부터 분리시켜 줌.          



### ◎ [스케치-사진 데이터를 짝으로 맺고 npz로 저장](https://github.com/maiorem/sketch_to_photo/blob/main/Project/pix2pix/pix2pix_0_save_new.py)           
<br />

## ▶ 모델 구성

### Generator(생성자) 모델 구성
#### Encoder
![encoder](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fee08Xh%2FbtqQ0PuuAVF%2FdwJgg2BvSz7ceGmSXkjjlk%2Fimg.png)     
 * 다운샘플링으로 쌓아 갈 레이어를 반환
<br /> 
#### Decoder
![decoder](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvmNJc%2FbtqQ7UnXkRN%2Fi4FcIe9cA9Ke9lkIKkSV71%2Fimg.png)      
 * 업샘플링으로 이미지 데이터를 디코딩 할 레이어를 반환.           
 이 때, 인코딩 레이어와 병합하여 데이터 손실을 막는다. (U-Net 구조)
<br />
#### U-Net 구조의 Generator
![generator](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FE9qHS%2FbtqQ0QmyDFp%2FekfexEFnG7wxtmEZVDXRk1%2Fimg.png)      

<br />

### Discriminator(판별자) 모델 구성
![discriminator](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHhPXb%2FbtqQ1MRRAbH%2FQUD52A1kdVFBHSGykk7PQk%2Fimg.png)    
 * patchGan으로 전체 데이터가 아닌 patch로 데이터의 진짜 가짜 여부를 판별하게 함.
<br />

### GAN 모델 구성
![gan](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd2Vu1O%2FbtqQXBQ1RWs%2FZ3rkruKjF3Klc3qo25kvy0%2Fimg.png)      

<br />

## ▶ 훈련
![train](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEFbmD%2FbtqQZxU5NDp%2FBG3KAANtBV0Aku51vIXfek%2Fimg.png)      

<br />

### [모델 구성부터 훈련까지 코드 전체 보기](https://github.com/maiorem/sketch_to_photo/blob/main/Project/pix2pix/pix2pix_00_main.py)
<br />

## ▶ 결과 확인

### 분류모델 구성
 * VGG16 전이학습 모델을 가져와서 sketch 모델을 라벨 별로 분류하고 모델 저장
 
[sketch 분류하기](https://github.com/maiorem/sketch_to_photo/blob/main/Project/pix2pix/vgg16.py)        
<br />

### Flask로 분류모델과 Pix2Pix모델 불러오기
 * 분류모델의 predict 값을 label과 각 카테고리별 확률을 둘 다 가져와서 HTML 파일로 전송
 * Pix2Pix모델의 predict 값을 디코딩하여 서버에 저장한 후 HTML로 출력

[FLASK 코드](https://github.com/maiorem/sketch_to_photo/blob/main/Project/flask_project/test_flask.py)           
<br />
 
## ▶ 시연
[Flask 시연 영상](https://youtu.be/9ftY67cGDOA)        
