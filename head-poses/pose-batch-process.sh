#!/usr/bin/bash

DEST="104.52.63.66"
PORT="7002"

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

flag=0
sus_count=0
total_count=0
for FNAME in ${FLIST[@]}; do
    echo -n "$FNAME "
    sleep 0.1
    curr=$( curl -X POST -F "file=@$FNAME" $DEST:$PORT 2>/dev/null | grep -oP "head_pose_angles.*}" )
    if [ $? -ne 0 ]; then curr="yaw0pitch0"; fi

    yaw00=$( echo $curr | grep -oP "yaw.*" | grep -oP "[-]?\\d+" | head -n1 )
    yaw=${yaw00#-} # absolute value for yaw only
    pitch=$( echo $curr | grep -oP "pitch.*" | grep -oP "[-]?\\d+" | head -n1 )

    # Consecutive deviation from centered position will be flagged
    status="OK"
    total_count=$(( $total_count + 1 ))

    if   [ $yaw -ge 60 ] || [ $pitch -ge 60 ] || [ $pitch -le -60 ]; then status="ignore"
    elif [ $yaw -ge 15 ] || [ $pitch -ge 25 ] || [ $pitch -le -10 ]; then
        flag=$(( $flag + 1 ))
        if [ $flag -gt 1 ]; then
            status="suspicious"
            sus_count=$(( $sus_count + 1 ))
        fi
    else flag=0
    fi

    printf "yaw %3d, pitch %3d, %s\n" $yaw00 $pitch $status
done

sus_percent=$(( $sus_count * 100 / $total_count ))
printf "%3d out of %3d frames suspicious\n%3d%% sus\n" $sus_count $total_count $sus_percent

