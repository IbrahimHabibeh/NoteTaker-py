# NoteTaker-py
Automatic notetaker for videos and audio recordings. The script can take any YouTube video and automatically take notes on it by extracting important sentences.


Note:
I am currently porting the project to C++ and making the project use an abstractive sentence extractor in order to improve the accuracy of the notetaker. Currently, the project only pulls important sentences and doesn't take the main point from each sentence. The project will also be on a win32 application, rather than a command line app. 


To Run the project do the following: 

1. install requirements.txt
pip install -r requirements.txt

2. Install extra packages

pip install bert-extractive-summarizer
pip install -U openai-whisper

3. Install nltk packages. Run: 

python
import nltk
nltk.download('punkt')

4. Run the project
python transcribe.py

Make sure to input a proper YouTube URL. Make sure the link doesn't include the timestamp. The best way to retrieve the link is to navigate to the YouTube video, right click on it, and press "Copy Video URL."
