## Downloading a fully setup docker image
For development, a x86 docker image was setup (11GB compressed)  
It has python environment setup and sourced to run the ML tasks  
It has the code repo checked out at /home/ubuntu/OnlyCode  
The bundled ssh key only have read-only permissions for git pull  

### Download and run
```
docker run -it -p 7001-7003:7001-7003 --name u24 alien222222/onlycode:x86
```
Sample output  
```
(ve) root@b8a9ed52dde0:/home/ubuntu# ls
OnlyCode  ve
```

### Stop
```
exit
docker stop u24
```

### Start
```
docker start u24
docker attach u24
```

### Info
```
docker stats
docker ps -a
docker images
```

### Commit
```
docker commit u24 alien222222/onlycode:x86
```

### Upload
```
docker push alien222222/onlycode:x86
```

### AWS Quick Start (no docker due to storage limit)
```
sudo apt update; sudo apt upgrade -y
sudo apt install -y ffmpeg git python3-pip tesseract-ocr
sudo apt install -y python3.12-venv
python3 -m venv /home/ubuntu/ve
echo "source /home/ubuntu/ve/bin/activate" >> /home/ubuntu/.bashrc
ssh-keygen -t ed25519 -f /home/ubuntu/.ssh/id_ed25519 -N ""
cat /home/ubuntu/.ssh/id_ed25519.pub
sudo reboot now

pip3 install mediapipe tf-keras transformers torch openai-whisper pytesseract flask waitress
git -c http.sslVerify=false clone git@github.com:flowac/OnlyCode.git; cd OnlyCode; git pull
(cd /home/ubuntu/OnlyCode/head-poses/; nohup python3 pose-service.py &)
(cd /home/ubuntu/OnlyCode/image/; nohup python3 count-service.py &)
(cd /home/ubuntu/OnlyCode/voice/; nohup python3 voice-service.py &)
```
