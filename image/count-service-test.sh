DEST="127.0.0.1"
PORT="7003"

FLIST=(data/*)

if [ $# -ge 1 ]; then
    DEST=$1
    if [ $# -ge 2 ]; then
        PORT=$2
    fi
fi

curl -X GET $DEST:$PORT
echo

for FNAME in ${FLIST[@]}; do
    sleep 0.1
    (set -x; curl -X POST -F "file=@$FNAME" $DEST:$PORT)
    echo
done

