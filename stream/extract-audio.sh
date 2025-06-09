video="terry.mp4"
if [ $# -gt 0 ]; then
    video=$1
fi

#time ffmpeg -i $video -vn -acodec mp3 $audio

segment=60
cutoff=30

mode="mp3"
ffprobe -v quiet -show_entries stream=codec_name $video | grep -q "codec_name=mp3"
if [ $? -eq 0 ]; then mode="copy"; fi

chop() {
    if [ $# -ne 2 ]; then exit; fi
    fname=$(printf "${video}_%02d.mp3" $2)
    (set -x; ffmpeg -v quiet -i $video -ss $1 -t $segment -y -vn -acodec $mode $fname)
}

idx=1
pos=0
rem=$(ffprobe -v quiet -show_entries format=duration $video | grep -oP \\d+ | head -n1)
while [ $rem -gt $segment ]; do
    rem=$(( $rem - $segment ))
    chop $pos $idx
    idx=$(( $idx +  1 ))
    pos=$(( $pos + $segment ))
done
# short segments will be truncated
if [ $rem -ge $cutoff ]; then
    chop $pos $idx
fi

