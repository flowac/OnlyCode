video="terry.mp4"
if [ $# -gt 0 ]; then
    video=$1
fi
audio="${video}.mp3"

rm -f $audio
time ffmpeg -i $video -vn -acodec mp3 $audio

