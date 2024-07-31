import streamlit as st
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

# Function to create animated text clips
def create_animated_text(text, duration, size):
    words = text.split()
    word_durations = duration / len(words)
    text_clips = []
    for i, word in enumerate(words):
        txt_clip = TextClip(word, fontsize=24, color='yellow', size=size, method='caption', bg_color='black', align='South')
        txt_clip = txt_clip.set_duration(word_durations).set_position(('center', 'bottom'))
        text_clips.append(txt_clip)
    return concatenate_videoclips(text_clips)

# Streamlit app
st.title("Video Editor")

# Number of scenes
num_scenes = st.number_input("Number of Scenes", min_value=1, max_value=10, value=5)

# Collect inputs for each scene
scenes = []
for i in range(num_scenes):
    st.header(f"Scene {i+1}")
    image_file = st.file_uploader(f"Upload Image for Scene {i+1}", type=["jpg", "jpeg", "png"])
    text_script = st.text_area(f"Text Script for Scene {i+1}")
    audio_file = st.file_uploader(f"Upload Audio for Scene {i+1}", type=["mp3", "wav"])
    image_animation = st.selectbox(f"Select Image Animation for Scene {i+1}", ["None", "Fade In", "Slide In"])
    text_animation = st.selectbox(f"Select Text Animation for Scene {i+1}", ["None", "Highlight Words"])

    if image_file and audio_file:
        scenes.append({
            "image": image_file,
            "text": text_script,
            "audio": audio_file,
            "image_animation": image_animation,
            "text_animation": text_animation
        })

# Generate video
if st.button("Generate Video"):
    final_clips = []
    for scene in scenes:
        image_clip = ImageClip(scene["image"].name).set_duration(AudioFileClip(scene["audio"].name).duration)
        audio_clip = AudioFileClip(scene["audio"].name)
        
        if scene["text_animation"] == "Highlight Words":
            subtitle_clip = create_animated_text(scene["text"], audio_clip.duration, image_clip.size)
        else:
            subtitle_clip = TextClip(scene["text"], fontsize=24, color='white', size=image_clip.size, method='caption', bg_color='black', align='South').set_duration(audio_clip.duration).set_position(('center', 'bottom'))
        
        final_clip = CompositeVideoClip([image_clip, subtitle_clip]).set_audio(audio_clip)
        final_clips.append(final_clip)

    # Add transitions between clips
    transition_duration = 1  # Duration of the transition in seconds
    final_clips_with_transitions = []
    for i in range(len(final_clips) - 1):
        final_clips_with_transitions.append(final_clips[i])
        transition = final_clips[i+1].crossfadein(transition_duration)
        final_clips_with_transitions.append(transition)
    final_clips_with_transitions.append(final_clips[-1])

    # Concatenate all clips with transitions
    final_video = concatenate_videoclips(final_clips_with_transitions, method="compose")

    # Export the final video
    final_video.write_videofile("yoga_video.mp4", codec="libx264", fps=24)
    st.success("Video generated successfully! Download it here.")
