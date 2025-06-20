# code sample rate is 1 frame every 10 seconds

video="lc004.mp4"
if [ $# -gt 0 ]; then
    video=$1
fi
prefix="${video}_%03d.jpg"

rm -f "${video}*jpg"
time ffmpeg -i $video -r 0.1 -qscale:v 5 $prefix

