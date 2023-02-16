import whisper
from pytube import YouTube
from summarizer import Summarizer
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

def get_clusters(text_data):
    n_clusters = 1; 
    scores = []
    n_clusters = 1
    while n_clusters < 10: 
        n_clusters += 1
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(text_data)
        if n_clusters == len(kmeans.labels_):
            break;
        scores.append(silhouette_score(text_data, kmeans.labels_))
    s_score = range(2, n_clusters + 1)[np.argmax(scores)]
    print("s_score: %s" % str(s_score))
    return s_score
def create_bullet_points(text):
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Tokenize the input text
    sentences = sent_tokenize(text)
    sentence_embeddings = st_model.encode(sentences) 
    # Perform clustering on the sentence embeddings
    text_data = np.array(sentence_embeddings)
    text_data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=get_clusters(text_data), random_state=0).fit(text_data)
    
    # Use the cluster labels to group similar sentences and add indentations
    bullet_points = []
    #r = sb_model.run_embeddings(text)
    for i in range(len(sentences)):
        indentation = "  " * kmeans.labels_[i]
        bullet_points.append(indentation + sentences[i])
    
    return bullet_points

video = YouTube(str(input("Input Link: "))).streams.filter(only_audio=True).first().download(filename="audio.wav")
print("fetching YouTube video...")
model = whisper.load_model("base.en")
print("Transcribing video...")
result = model.transcribe("audio.wav")
print("Extracting important points from video...")
s_model = Summarizer()
notes = s_model(result['text'])
print("Creating a bullet point list...")
with open("out/output.txt", "w") as file: 
    bullet_points = create_bullet_points(notes)
    for point in bullet_points: 
        file.write(point + '\n')
    file.close()
with open("out/transcription.txt", "w") as file: 
    file.write(result['text'])
    file.close()

print(notes)