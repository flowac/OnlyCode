#!/usr/bin/bash

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
flag=0
sus_count=0
total_count=0
for FNAME in ${FLIST[@]}; do
    echo -n "$FNAME "
    sleep 0.1
    curr=$(curl -X POST -F "file=@$FNAME" $DEST:$PORT 2>/dev/null | grep -oP \\d+)
    if [ $? -ne 0 ]; then curr=0; fi
    diff=$(( $curr - $prev ))

    # Sampling 1 frame every 10 seconds
    # Effective words per minute is 6 times diff
    # 0 wpm or less, or greater than 180 wpm will be ignored
    # Consecutive higher than or equal to 42 wpm will be flagged
    status="OK"
    total_count=$(( $total_count + 1 ))

    if [ $diff -le 0 ] || [ $diff -ge 30 ]; then status="ignore"
    elif [ $diff -ge 7 ]; then
        flag=$(( $flag + 1 ))
        if [ $flag -gt 1 ]; then
            status="suspicious"
            sus_count=$(( $sus_count + 1 ))
        fi
    else flag=0
    fi

    printf "%3d %3d %s\n" $curr $diff $status
    prev=$curr
done

sus_percent=$(( $sus_count * 100 / $total_count ))
printf "%3d out of %3d frames suspicious\n%3d%% sus\n" $sus_count $total_count $sus_percent

