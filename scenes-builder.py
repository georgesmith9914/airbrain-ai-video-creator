from moviepy.editor import *
from moviepy.config import change_settings
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import cv2
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

from dotenv import load_dotenv

load_dotenv()

create_full_video = True
generate_media_files = True

prompt_to_generate_summary = "I need to create a video to summarize key lessons from  4 hours work week book. it wil have 5 scenes. Each scene has 1 sentence about the book as the script. Now generate the script for 5 scenes, 1 sentence each."
prompt_to_generate_intro = "also make 1 line intro to this summary video"

scenes = [
    {
        "image": "./images/1-generated_image.png",
        "text": "Discover the secrets to a more productive and fulfilling life with key insights from ‘The 4-Hour Workweek’ by Tim Ferriss."
    },
    {
        "image": "./images/2-generated_image.png",
        "text": "Escape the 9-5 grind by focusing on what truly matters and eliminating unnecessary tasks."
    },
    {
        "image": "./images/3-generated_image.png",
        "text": "Automate income streams to create a lifestyle of financial freedom and flexibility."
    },
    {
        "image": "./images/4-generated_image.png",
        "text": "Embrace the concept of mini-retirements to enjoy life now, not just in the distant future."
    },
    {
        "image": "./images/5-generated_image.png",
        "text": "Escape the 9-5 grind by focusing on what truly matters and eliminating unnecessary tasks."
    },
    {
        "image": "./images/6-generated_image.png",
        "text": "Redefine success by prioritizing personal fulfillment and experiences over material wealth."
    }
]

def generate_audio(content, sceneId):
    import azure.cognitiveservices.speech as speechsdk
    speech_key, service_region = os.getenv("AZURE_SPEECH_API_KEY"), os.getenv("AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    voice = os.getenv("AZURE_SPEECH_VOICE")
    speech_config.speech_synthesis_voice_name = voice
    filename = str(scene_id) + "-" + "audio.mp4"
    audio_config = speechsdk.AudioConfig(filename=filename)

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_synthesizer.speak_text_async(content).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:

        print(f"Audio saved to {filename}")
        return filename

    else:

        print(f"Error: {result.error_details}")

def create_animated_subtitles(text, duration, size):
    words = text.split()
    word_durations = duration / len(words)
    text_clips = []
    for i, word in enumerate(words):
        #print(size)
        txt_clip = TextClip(word, fontsize=100, color='yellow', size=(0,0), stroke_color='black', stroke_width=2) 
        txt_clip = txt_clip.set_duration(word_durations)
        text_clips.append(txt_clip)
    return concatenate_videoclips(text_clips, method="compose")

def Zoom(clip,mode='in',position='center',speed=1):
    fps = 24
    duration = clip.duration
    total_frames = int(duration*fps)
    def main(getframe,t):
        frame = getframe(t)
        h,w = frame.shape[:2]
        i = t*fps
        if mode == 'out':
            i = total_frames-i
        zoom = 1+(i*((0.1*speed)/total_frames))
        positions = {'center':[(w-(w*zoom))/2,(h-(h*zoom))/2],
                     'left':[0,(h-(h*zoom))/2],
                     'right':[(w-(w*zoom)),(h-(h*zoom))/2],
                     'top':[(w-(w*zoom))/2,0],
                     'topleft':[0,0],
                     'topright':[(w-(w*zoom)),0],
                     'bottom':[(w-(w*zoom))/2,(h-(h*zoom))],
                     'bottomleft':[0,(h-(h*zoom))],
                     'bottomright':[(w-(w*zoom)),(h-(h*zoom))]}
        tx,ty = positions[position]
        M = np.array([[zoom,0,tx], [0,zoom,ty]])
        frame = cv2.warpAffine(frame,M,(w,h))
        return frame
    return clip.fl(main)

# Create a list to store all the video clips
video_clips = []

# Iterate over each scene
for scene in scenes:
    scene_id = scenes.index(scene) + 1
    # Load the image
    image = ImageClip(scene["image"])

    # Load the audio
    audio = None
    if(generate_media_files):
        audio = AudioFileClip(generate_audio(scene["text"], scene_id))
    else:
        audio = AudioFileClip(str(scene_id) + "-audio.mp4")

    subtitle_clip = create_animated_subtitles(scene["text"], audio.duration, image.size)
    subtitle_clip = subtitle_clip.set_position(("center", 800))


    # Combine image and text
    video = CompositeVideoClip([image, subtitle_clip])
    # Set the duration of the video to match the audio
    video = video.set_duration(audio.duration)
    video = video.set_fps(24)

    final_clip_with_fadein = video.crossfadein(1)
    # Apply the Zoom effect with alternating mode
    #final_clip = Zoom(clip=video_clip, mode='in' if i % 2 == 0 else 'out', position='center', speed=1.2)
    video = Zoom(clip=final_clip_with_fadein, mode='in' if scene_id % 2 == 0 else 'out', position='center', speed=1.2)

    # Set the audio to the video
    video = video.set_audio(audio)



    # Append the video clip to the list
    video_clips.append(video)

    # Save the video clip to a file
    video.write_videofile(f"scene_video_{scene_id}.mp4", fps=24)

if(create_full_video):
    # Concatenate all the video clips
    final_video = concatenate_videoclips(video_clips)

    background_music = AudioFileClip("./bg_music.mp3")
    original_audio = final_video.audio
    background_music = background_music.subclip(0, final_video.duration)
    combined_audio = CompositeAudioClip([original_audio, background_music.volumex(0.5)])
    final_video = final_video.set_audio(combined_audio)


    # Save the final video to a file
    final_video.write_videofile("final_video.mp4", codec="libx264", fps=24)

