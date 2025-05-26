set -x
DEST="127.0.0.1"
PORT="7001"

if [ $# -ge 1 ]; then
    DEST=$1
    if [ $# -ge 2 ]; then
        PORT=$2
    fi
fi

sleep 0.2; curl -X GET $DEST:$PORT
sleep 0.2; curl -X POST -F "file=@data/terry.mp3" $DEST:$PORT
sleep 0.2; curl -X POST -F "file=@data/micro-machine.wav" $DEST:$PORT
sleep 0.2; curl -X POST -F "file=@data/badname" $DEST:$PORT
sleep 0.2; curl -X POST -F "file=@data/bad.wav" $DEST:$PORT

