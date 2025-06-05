## Downloading a fully setup docker image
For development, a x86 docker image was setup (11GB compressed)  
It has python environment setup and sourced to run the ML tasks  
It has the code repo checked out at /home/ubuntu/OnlyCode  
The bundled ssh key only have read-only permissions for git pull  

### Download and run
```
docker run -it --name u24 alien222222/onlycode:x86
```
Sample output  
```
$ docker run -it --name u24 alien222222/onlycode:x86
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

