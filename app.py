import os
from flask import Flask, request
from dotenv import load_dotenv
import create_video as cv
print(cv)

app = Flask(__name__)
port = int(os.environ.get('PORT', 3000))

load_dotenv()



@app.route('/create_video')
def create_video():
    topic=request.args.get("topic")
    print(topic)
    print("Step 1: Generate the content")
    content = cv.generate_content(topic, 5)
    print("Content:", content)

    print("Step 2: Summarize the content")
    num_of_sentences = 3
    summary = ""
    summary = cv.summarize_content(content, num_of_sentences)
    print("Summary:", summary)

    print("Step 3: Extract key phrases")
    phrase_list = []
    phrases = ""
    phrase_list, phrases = cv.extract_key_phrases(summary)
    print("Key Phrases:", phrase_list)
    print("Phrases:", phrases)

    print("Step 4: Generate images")
    max_number_of_phrases = 4
    cv.generate_images(phrase_list, max_number_of_phrases)
    print("Images generated.....")

    print("Step 5: Generate audio")
    cv.generate_audio(summary)
    print("Audio generated.....")

    print("Step 6: Generate video")
    audio_fileName = "audio.mp4"
    cv.stitch_video(phrase_list, max_number_of_phrases, audio_fileName)
    print("Video generated.....")

    return "video.mp4"
    #return result["processed_result"]
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)