# EverybodyDanceNow reproduced in pytorch

## 아래 open source code 참조:

[nyoki-mtl](https://github.com/CUHKSZ-TQL/EverybodyDanceNow_reproduce_pytorch) EverybodyDanceNow reproduced in pytorch


## Full process
현재 여기 있는 모델은 재명이 형 동영상으로 만든 generative model
![](/results/target_bruno_mars/output_facegan.gif)

## 필요한 pre-trained model
* Download vgg19-dcbb9e9d.pth.crdownload [here](https://drive.google.com/file/d/1JG-pLXkPmyx3o4L33rG5WMJKMoOjlXhl/view?usp=sharing) and put it in `./src/pix2pixHD/models/`  <br>

* Download pose_model.pth [here](https://drive.google.com/file/d/1DDBQsoZ94N4NRKxZbwyEXt7Tz8KqgS_w/view?usp=sharing) and put it in `./src/PoseEstimation/network/weight/`   <br>

* Download pre-trained vgg_16 for face enhancement [here](https://drive.google.com/file/d/180WgIzh0aV1Aayl_b1X7mIhVhDUcW3b1/view?usp=sharing) and put in `./face_enhancer/`

#### Training 방법
* target video(generative model의 학습 비디오)에 대해 `mv.mp4`로 이름 변경 후, `./data/target/`에 넣고 `python make_target.py` 실행
* `python train_pose2vid.py` 실행하여 학습 후, pose2vid 모델  `./checkpoints/`에 저장
* 학습을 멈추고 최근 학습된 모델에서 initialization 하여 재학습 하고 싶은 경우, `./src/config/train_opt.py`파일에서 `load_pretrain = './checkpoints/target/'`로 수정
* `cd face_enhancer` 로 face_enhancer 디렉토리로 이동 후, `python prepare.py` 실행 `./data/face/test_sync`와 `./data/face/test_real` 생성 확인 가능
* `python main.py` 실행하여 학습 후, face gan 모델 `./checkpoints/`에 저장

#### Genrate 방법(Test)
* source video(Generative model의 input)에 대해  `mv.mp4`로 이름 변경 후,  `./data/source/`에 넣고 `python make_source.py` 실행`./data/source/test_label_ori`와 `./data/source/pose_souce.npy` 생성 확인 가능
* `python normalization.py` 실행하여 target video에 맞게 pose를 rescale
* `python transfer.py` 실행 하여 `./result`에서 결과 확인
* `cd face_enhancer`로 face_enhancer 디렉토리로 이동 후, `python enhance.py` 실행
* `cd ..`한 후, `python make_gif.py` 실행하여 gif 파일 생성
* 메모리 문제로 위 실행이 안될 수 있음 이경우, `jupyter notebook` 실행하여 gif 생성
 
## TODO
- Pose estimation
    - [x] Pose
    - [x] Face
    - [ ] Hand
- [x] pix2pixHD
- [x] FaceGAN
- [ ] Temporal smoothing

