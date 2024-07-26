#move all secrets to .env
#handle content filter exceptions for image generation
import os

import openai


from dotenv import load_dotenv

load_dotenv()


import os
import openai
from openai import AzureOpenAI
from moviepy.editor import *
from PIL import Image
import glob
import cv2
import numpy as np
import requests
from moviepy.editor import ImageClip, CompositeVideoClip, TextClip
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})



def generate_content(topic, num_of_sentences):
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = 'Summarize the key points about '+ topic + " in " + str(num_of_sentences) +' lines.' 
    print(prompt)

    client = AzureOpenAI(
        api_key=openai.api_key,
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")
    completion = client.chat.completions.create(
      model=deployment_name,
      messages=[
        {"role": "system", "content": "You are a helpful assistant that helps summarize"},
        {"role": "user", "content": prompt}
      ]
    )
    text = completion.choices[0].message.content
    print("Assistant: " + text)
    return text

def summarize_content(content, num_of_sentences):
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    openai.api_key = os.getenv("OPENAI_API_KEY")


    prompt = 'Provide a summary of the text below that captures its main idea in '+ str(num_of_sentences) +' sentences. Tell this in a conversational way that is easy to understand for a general audience. It should energize the audience. \n' + content 
    print(prompt)

    client = AzureOpenAI(
        api_key=openai.api_key,
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")
    completion = client.chat.completions.create(
      model=deployment_name,
      messages=[
        {"role": "system", "content": "You are a helpful assistant that helps summarize"},
        {"role": "user", "content": prompt}
      ]
    )
    response_sum = completion.choices[0].message.content
    print("Assistant: " + response_sum)
    return response_sum



def authenticate_text_analytics_client():
    key = os.getenv("AZURE_LANGUAGE_API_KEY")

    endpoint = os.getenv("AZURE_LANGUAGE_API_ENDPOINT")

    from azure.ai.textanalytics import TextAnalyticsClient

    from azure.core.credentials import AzureKeyCredential
    
    ta_credential = AzureKeyCredential(key)

    text_analytics_client = TextAnalyticsClient(

            endpoint=endpoint,

            credential=ta_credential)

    return text_analytics_client





def extract_key_phrases(summarized_content):

    try:

        client = authenticate_text_analytics_client()

        phrase_list, phrases = [], ''

        documents = [summarized_content]



        response_kp = client.extract_key_phrases(documents = documents)[0]



        if not response_kp.is_error:

            print("\tKey Phrases:")

            for phrase in response_kp.key_phrases:

                print("\t\t", phrase)

                phrase_list.append(phrase)

                phrases = phrases +"\n"+ phrase          
            return phrase_list, phrases        
        else:

            print(response_kp.id, response_kp.error)



    except Exception as err:

        print("Encountered exception. {}".format(err))

    return phrase_list, phrases



def get_image_phrases():

    phrase_list, phrases = extract_key_phrases()

    prompt = ''' Provide an image idea for each phrases: ''' + phrases
    print(prompt)

    import openai
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key = openai.api_key,
        azure_endpoint = os.getenv("OPENAI_API_BASE"),
        api_version = os.getenv("OPENAI_API_VERSION")
        
    )
    deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")
    # Assuming 'openai.api_key' is set elsewhere in your code or environment variables
    completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that helps provide ideas for images based on key phrases. Do not use numbers for the list. Rewrite the prompt to be simpler and avoid any references to NSFW content, copyrighted characters, or controversial topics. When mentioning art styles, only include artists whose work predates 1912, or describe the style in general terms. Ensure the idea focuses on safe, universally acceptable themes without explicit, violent, or inappropriate content. Avoid political, controversial, or sensitive issues that might provoke or offend."},
        {"role": "user", "content": prompt}
    ]
    )
    print("Assistant: " + completion.choices[0].message.content)
    response_phrase = completion.choices[0].message.content + ""

    image_phrases = response_phrase.split("\n")[1:]
    print(image_phrases)
    return image_phrases

def fetch_and_prepare_images(phrases, max_number_of_phrases):
    print("Fetching and preparing images.....")
    prompt = ''' Provide an image idea for each phrases: ''' + phrases
    print(prompt)

    import openai
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key = openai.api_key,
        azure_endpoint = os.getenv("OPENAI_API_BASE"),
        api_version = os.getenv("OPENAI_API_VERSION")
        
    )
    deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")
    # Assuming 'openai.api_key' is set elsewhere in your code or environment variables
    completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that helps provide ideas for images based on key phrases. Do not use numbers for the list. Rewrite the prompt to be simpler and avoid any references to NSFW content, copyrighted characters, or controversial topics. When mentioning art styles, only include artists whose work predates 1912, or describe the style in general terms. Ensure the idea focuses on safe, universally acceptable themes without explicit, violent, or inappropriate content. Avoid political, controversial, or sensitive issues that might provoke or offend."},
        {"role": "user", "content": prompt}
    ]
    )
    print("Assistant: " + completion.choices[0].message.content)
    response_phrase = completion.choices[0].message.content + ""

    image_phrases = response_phrase.split("\n")[1:]
    print(image_phrases)



    # Base URL for the API call
    base_url = os.getenv("PEXELS_IMAGES_API_BASE")

    # Your API key
    api_key = os.getenv("PEXELS_API_KEY")

    # Headers for authorization
    headers = {
        "Authorization": api_key
    }

    image_counter = 0
    # Loop through each phrase in the list
    for phrase in image_phrases[:max_number_of_phrases]:
        # Format the query parameter with the current phrase
        params = {
            "query": phrase,
            "per_page": 1
        }
        
        # Make the API call
        response = requests.get(base_url, headers=headers, params=params)

        #print(response)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Extract the image URL from the response
            data = response.json()
            if data['photos']:
                image_url = data['photos'][0]['src']['original']
                # Download the image
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    # Extract the file extension from the image URL
                    _, extension = os.path.splitext(image_url)
                    # Ensure the extension is not empty and has a valid image format; otherwise, default to '.jpg'
                    if not extension in ['.jpg', '.jpeg', '.png', '.gif']:
                        extension = '.jpg'
                    # Save the image to a file with the correct extension
                    #image_path = f"./images/{phrase.replace(' ', '_')}{extension}"
                    image_path = f"./images/{image_counter}-generated_image{extension}"
                    with open(image_path, 'wb') as file:
                        file.write(image_response.content)
                        # Open the JPEG file and convert it to RGB
                    if extension == '.jpg' or extension == '.jpeg':
                            # Open the PNG file and convert it to RGB
                            im = Image.open(image_path).convert("RGB")
                            # Create a .jpeg filename by replacing the .png extension
                            png_file = image_path.rsplit('.', 1)[0] + '.png'
                            # Save the image as a .jpeg
                            im.save(png_file, "PNG")
                            # Optionally, delete the original PNG file
                            os.remove(image_path)
                            image_path = png_file  # Update image_path to point to the new JPEG file

                    print(f"Downloaded '{phrase}' image to {image_path}")
                    image_counter += 1
                else:
                    print(f"Failed to download image for '{phrase}'")
            else:
                print(f"No photos found for '{phrase}'")
        else:
            print(f"Failed to fetch data for '{phrase}'")

    # List all .webp files in the current directory
    #webp_files = glob.glob('./prepare_images/*.webp')

    #for webp_file in webp_files:
        # Open the .webp image
    #    im = Image.open(webp_file).convert("RGB")
    #    # Create a .jpeg filename by replacing the .webp extension
    #    jpeg_file = webp_file.rsplit('.', 1)[0] + '.jpg'
        # Save the image as a .jpeg
    #    im.save(jpeg_file, "JPEG")

def generate_images(image_phrases, max_number_of_phrases):
    import requests

    import time

    import os

    api_base = os.getenv("OPENAI_API_BASE")

    api_key = os.getenv("OPENAI_API_KEY")

    api_version = os.getenv("OPENAI_API_VERSION")


    url = "{}openai/deployments/dall-e-3/images/generations?api-version={}".format(api_base, api_version)

    headers= { "api-key": api_key, "Content-Type": "application/json" }

    import requests
    import time
    import json
    from PIL import Image

    print(os.getenv("IMAGE_MODEL_API_ENDPOINT"))
    # Assuming 'url' and 'headers' are defined above this snippet
    images = []  # Ensure this is defined if you're collecting results
    client = AzureOpenAI(
        api_version=os.getenv("IMAGE_MODEL_API_VERSION"),  
        api_key=os.getenv("OPENAI_API_KEY"),  
        azure_endpoint=os.getenv("IMAGE_MODEL_API_ENDPOINT")   
    )
    deployment_name = os.getenv("IMAGE_MODEL_DEPLOYMENT_NAME")
    imgCount = 0

    try:
        for i, phrase in enumerate(image_phrases):
            if i >= max_number_of_phrases:
                break
            print(f"Generating image for phrase: {phrase}")
            if phrase != "":
                result = client.images.generate(
                    model=os.getenv("IMAGE_MODEL_NAME"),  # the name of your DALL-E 3 deployment
                    prompt=phrase,
                    n=1
                )

                json_response = json.loads(result.model_dump_json())

                # Set the directory for the stored image
                image_dir = os.path.join(os.curdir, 'images')

                # If the directory doesn't exist, create it
                if not os.path.isdir(image_dir):
                    os.mkdir(image_dir)

                # Initialize the image path (note the filetype should be png)
                image_path = os.path.join(image_dir, str(imgCount) + '-generated_image.png')
                imgCount += 1

                # Retrieve the generated image
                image_url = json_response["data"][0]["url"]  # extract image URL from response
                generated_image = requests.get(image_url).content  # download the image
                with open(image_path, "wb") as image_file:
                    image_file.write(generated_image)

                # Display the image in the default image viewer
                image = Image.open(image_path)
                #image.show()
    except Exception as e:
        print("Error generating images")
        print(e)


def generate_audio(content):
    import azure.cognitiveservices.speech as speechsdk
    speech_key, service_region = os.getenv("AZURE_SPEECH_API_KEY"), os.getenv("AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    voice = os.getenv("AZURE_SPEECH_VOICE")
    speech_config.speech_synthesis_voice_name = voice
    filename = "audio.mp4"
    audio_config = speechsdk.AudioConfig(filename=filename)

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_synthesizer.speak_text_async(content).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:

        print(f"Audio saved to {filename}")

    else:

        print(f"Error: {result.error_details}")



#Stich the audio files and the images together

def Zoom(clip,mode='in',position='center',speed=1):
    fps = clip.fps
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

def divide_text_into_parts(text, num_parts=10):
    # Split the text into words
    words = text.split()
    
    # Calculate the number of words per part
    words_per_part = len(words) / num_parts
    
    # Initialize variables to hold the current count and the result
    current_count = 0
    parts = []
    current_part = []
    
    # Iterate over the words and divide them into parts
    for word in words:
        current_part.append(word)
        current_count += 1
        
        # Check if the current part has reached the approximate size
        if current_count >= words_per_part:
            parts.append(' '.join(current_part))
            current_part = []
            current_count = 0
    
    # Add the last part if there are any remaining words
    if current_part:
        parts.append(' '.join(current_part))
    
    # Adjust the last parts if there are fewer than num_parts due to rounding
    while len(parts) < num_parts:
        parts.append("")
    
    return parts

def create_video(images, audio, output, image_phrases, summary):
    from PIL import Image
    from PIL import Image as pil
    from pkg_resources import parse_version

    if parse_version(pil.__version__)>=parse_version('10.0.0'):
        Image.ANTIALIAS=Image.LANCZOS
    import textwrap


    print("Creating the video.....")


    #clips = [ImageClip(m).resize(height=1024).set_duration(5) for m in images]
    #clips = [Zoom(clip=ImageClip(m).resize(height=1024).set_duration(4).set_fps(30), mode='in', position='center', speed=1.2) for m in images]    
    #clips = [Zoom(clip=ImageClip(m).resize(height=1024).set_duration(4).set_fps(30), mode='in' if i % 2 == 0 else 'out', position='center', speed=1.2) for i, m in enumerate(images)]
    #print(clips)

    # Assuming 'texts' is a list of text strings corresponding to each image in 'images'
    clips = []
    #texts = image_phrases
    texts = divide_text_into_parts(summary, len(images)-1)
    print(texts)
    for i, (m, text) in enumerate(zip(images, texts)):
        # Create an image clip
        img_clip = ImageClip(m).resize(height=1024).set_duration(2.95).set_fps(30)
        
        # Create a text clip (customize as needed)
        #txt_clip = TextClip(text, fontsize=70, color='white', font="Amiri-Bold", align='center').set_position('center').set_duration(4)
        # Enhanced text clip with shadow for better readability and animated fade-in
        #txt_clip = TextClip(text, fontsize=70, color='white', font="Arial-Bold", align='center').set_position(('center', 'bottom')).set_duration(4).margin(bottom=20, opacity=0)  # Add some bottom margin
        
        # Wrap text to a list of lines, where each line has up to 40 characters
        wrapped_text_lines = textwrap.wrap(text, width=40)
        # Join the list of lines into a single string with newline characters
        wrapped_text = "\n".join(wrapped_text_lines)

        # Create a more attractive text clip
        txt_clip = (TextClip(wrapped_text, fontsize=50, color='white', font="Roboto-Condensed", align='center', stroke_color='black', stroke_width=1)
                    .set_position(('center', 'center'))
                    .set_duration(2)
                    .margin(bottom=20, opacity=0)  # Add some bottom margin
                    .fadein(1)  # Apply fade-in effect for 1 second at the start
                    .fadeout(1))  # Apply fade-out effect for 1 second at the end
                    #.set_shadow(color='gray', offset=(1,1), transparency=0.7))  # Add a soft shadow for a nicer look
        # Overlay the text on the image
        video_clip = CompositeVideoClip([img_clip, txt_clip])
        # Apply fade-in effect to the entire composite clip
        final_clip_with_fadein = video_clip.crossfadein(1)
        # Apply the Zoom effect with alternating mode
        #final_clip = Zoom(clip=video_clip, mode='in' if i % 2 == 0 else 'out', position='center', speed=1.2)
        final_clip = Zoom(clip=final_clip_with_fadein, mode='in' if i % 2 == 0 else 'out', position='center', speed=1.2)
        
        clips.append(final_clip)

    concat_clip = concatenate_videoclips(clips, method="compose")

    audio_clip = AudioFileClip(audio)

    print("Writing to video.....1")

    final_clip = concat_clip.set_audio(audio_clip)

    print("Writing to video.....2")

    final_clip.write_videofile(output, fps=24)

def stitch_video(image_phrases, max_number_of_phrases, filename, summary):
    print(max_number_of_phrases)

    images = [f"images/{index}-generated_image.png" for index, _ in enumerate(image_phrases) if index < max_number_of_phrases]
    print(images)

    audio = filename

    print(audio)

    output = "video.mp4"

    create_video(images, audio, output, image_phrases, summary)
    print("Video created.....")
    return output

def main():
    print("Step 1: Generate the content")
    content = ""
    #Topics: Question, Books, Movies, Music, Health, Technology, Science, History, Art, Food, Travel, Research papers
    content = generate_content("Microsoft Autogen", 5)
    print("Content:", content)

    print("Step 2: Summarize the content")
    num_of_sentences = 3
    summary = "Hey there! You won't believe how amazing sleep can be for us. It's like a superhero, keeping our heart, blood vessels, and brain in tip-top shape, and even helping us maintain a healthy weight. But beware, if we skimp on it, it can cause stress and anxiety, reduce our productivity and even put our safety at risk! So, let's make sure we're getting the sleep we need to stay on top of our game."
    summary = summarize_content(content, num_of_sentences)
    print("Summary:", summary)

    print("Step 3: Extract key phrases")
    phrase_list = ['Manokamana Siddhi', 'fascinating concept', 'Hindu spirituality', 'personal wishes', 'sincere worship', 'spiritual pr actices', 'mantra chanting', 'beautiful belief', 'little bit', 'divine grace', 'selfless desires', 'fulfillment', 'meditation', 'fa ith', 'self-improvement', 'pursuit', 'positive', 'principles', 'Karma', 'mind', 'reality']
    phrases = ""
    phrase_list, phrases = extract_key_phrases(summary)
    print("Key Phrases:", phrase_list)
    print("Phrases:", phrases)

    print("Step 4a: Generate images")
    max_number_of_phrases = 15
    generate_images(phrase_list, max_number_of_phrases)
    print("Images generated.....")

    print("Step 4b: Fetch and Prepare images")
    max_number_of_phrases = 15
    #fetch_and_prepare_images(phrases, max_number_of_phrases)
    print("Images prepared.....")

    print("Step 5: Generate audio")
    generate_audio(summary)
    print("Audio generated.....")

    print("Step 6: Generate video")
    audio_fileName = "audio.mp4"
    stitch_video(phrase_list, max_number_of_phrases, audio_fileName, summary)
    print("Video generated.....")

if __name__ == "__main__":
    main()