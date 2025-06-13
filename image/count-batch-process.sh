DEST="104.52.63.66"
PORT="7003"

if [ $# -lt 1 ]; then exit; fi
FLIST=("$1"*)
echo "${#FLIST[@]} files starting with $1"
if [ ${#FLIST[@]} -lt 1 ]; then exit; fi

if [ $# -ge 2 ]; then DEST=$2; fi
if [ $# -ge 3 ]; then PORT=$3; fi

curl -X GET $DEST:$PORT 2>/dev/null | grep -q ready
if [ $? -ne 0 ]; then
    echo "Server $DEST:$PORT not ready"
    exit
fi

prev=0
for FNAME in ${FLIST[@]}; do
    echo -n "$FNAME "
    sleep 0.1
    curr=$(curl -X POST -F "file=@$FNAME" $DEST:$PORT 2>/dev/null | grep -oP \\d+)
    if [ $? -ne 0 ]; then curr=0; fi
    diff=$(( $curr - $prev ))

    status="ok"
    if [ $diff -le 0 ] || [ $diff -ge 30 ]; then status="-"
    elif [ $diff -ge 10 ]; then status="suspicious"
    fi

    printf "%3d %3d %s\n" $curr $diff $status
    prev=$curr
done

