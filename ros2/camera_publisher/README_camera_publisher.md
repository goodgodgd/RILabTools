# Camera setting

1. Droidcam
- 스마트폰 이미지를 1920 1080 해상도로 linux(server)로 전송할 수 있는 camera app 필요로 droidcam 선택
> droidcam이 적합하다 생각하여 선택 하였지만 다른 프로그램을 사용하여도 가능
- install 이 필요시 https://www.dev47apps.com/droidcam/linux/ 사이트 이용
- 설치 완료 됬을 시 부팅 시마다 다음 명령어 실행 필요
```
$ sudo rmmod v4l2loopback_dc
$ sudo insmod /lib/modules/`uname -r`/kernel/drivers/media/video/v4l2loopback-dc.ko width=1920 height=1080
```
> samsung 5g 노트북에는 다음과 같이 alias 설정되있어서 다음 명령어로도 실행 가능
```
$ cam_init
$ set_resolution
```
- droidcam 실행
```
$ droidcam
```


