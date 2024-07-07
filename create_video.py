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



def create_video(images, audio, output):
    from PIL import Image
    from PIL import Image as pil
    from pkg_resources import parse_version

    if parse_version(pil.__version__)>=parse_version('10.0.0'):
        Image.ANTIALIAS=Image.LANCZOS



    print("Creating the video.....")


    clips = [ImageClip(m).resize(height=1024).set_duration(5) for m in images]

    print(clips)

    concat_clip = concatenate_videoclips(clips, method="compose")

    audio_clip = AudioFileClip(audio)

    print("Writing to video.....1")

    final_clip = concat_clip.set_audio(audio_clip)

    print("Writing to video.....2")

    final_clip.write_videofile(output, fps=24)

def stitch_video(image_phrases, max_number_of_phrases, filename):
    print(len(image_phrases))

    images = [f"images/{index}-generated_image.png" for index, _ in enumerate(image_phrases) if index < max_number_of_phrases]
    print(images)

    audio = filename

    print(audio)

    output = "video.mp4"

    create_video(images, audio, output)
    print("Video created.....")
    return output

def main():
    print("Step 1: Generate the content")
    content = generate_content("Healthy Eating", 5)
    print("Content:", content)

    print("Step 2: Summarize the content")
    num_of_sentences = 3
    summary = ""
    summary = summarize_content(content, num_of_sentences)
    print("Summary:", summary)

    print("Step 3: Extract key phrases")
    phrase_list = []
    phrases = ""
    phrase_list, phrases = extract_key_phrases(summary)
    print("Key Phrases:", phrase_list)
    print("Phrases:", phrases)

    print("Step 4: Generate images")
    max_number_of_phrases = 4
    generate_images(phrase_list, max_number_of_phrases)
    print("Images generated.....")

    print("Step 5: Generate audio")
    generate_audio(summary)
    print("Audio generated.....")

    print("Step 6: Generate video")
    audio_fileName = "audio.mp4"
    stitch_video(phrase_list, max_number_of_phrases, audio_fileName)
    print("Video generated.....")

if __name__ == "__main__":
    main()