import io
import os
import requests
import json
from PIL import Image
import datasets

API_KEY = os.environ.get("YOUTUBE_API_KEY")
BASE_URL = 'https://www.googleapis.com/youtube/v3'

def get_title_thumbnail_pairs(channel_url, max_results=5):
    # Extract channel ID from URL
    channel_name = channel_url.split("/@")[-1]
    channel_id_url = f'{BASE_URL}/search?part=snippet&q={channel_name}&type=channel&key={API_KEY}'
    response = requests.get(channel_id_url, timeout=5)
    data = response.json()    

    # Extract the channel ID
    channel_id = data['items'][0]['snippet']['channelId']
    response = requests.get(f'{BASE_URL}/channels?part=contentDetails&id={channel_id}&key={API_KEY}')
    channel_data = response.json()

    # Extract the playlist ID
    playlist_id = channel_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    
    response = requests.get(f'{BASE_URL}/playlistItems?part=snippet&playlistId={playlist_id}&maxResults={max_results}&key={API_KEY}')
    videos = response.json()
    
    title_thumbnail_pairs = []

    for video in videos['items']:
        title = video['snippet']['title']
        thumbnail = video['snippet']['thumbnails']['high']['url']
        title_thumbnail_pairs.append((thumbnail, title))

    return title_thumbnail_pairs
    

def load_thumbnail(thumbnail_url):
    response = requests.get(thumbnail_url, timeout=5)
    response.raise_for_status() 
    img_bytes = response.content
    image = Image.open(io.BytesIO(img_bytes))    
    return image


#may need to change file_name to image https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
def create_dataset(channel_url, max_results=5):
    # title_thumbnail_pairs = get_title_thumbnail_pairs(channel_url, max_results)
    title_thumbnail_pairs = [('https://i.ytimg.com/vi/tWP6z0hvw1M/hqdefault.jpg', 'OpenAI shocks the world yet again… Sora first look'), ('https://i.ytimg.com/vi/DuqLkG75BE8/hqdefault.jpg', 'Zuck’s brutal takedown of Apple Vision Pro'), ('https://i.ytimg.com/vi/X8LglXSG53A/hqdefault.jpg', 'how god programmed birds probably'), ('https://i.ytimg.com/vi/4Wa5DivljOM/hqdefault.jpg', "this is why you're addicted to cloud computing"), ('https://i.ytimg.com/vi/ucd63nIZZ60/hqdefault.jpg', 'Google actually beat GPT-4 this time? Gemini Ultra released')]
    
    channel_name = channel_url.split("/@")[-1]
    dataset_name = f"{channel_name}_dataset/metadata.json"
    os.makedirs(dataset_name, exist_ok=True)

    json_data = []
    for i, (thumbnail_url, title) in enumerate(title_thumbnail_pairs):
        image = load_thumbnail(thumbnail_url)
        image.save(f"{dataset_name}/{i}.jpg")        
        json_data.append({"file_name": f"{i}.jpg", "text": title})
    
    with open(f"{dataset_name}/metadata.jsonl", "w") as f:
        for item in json_data:
            f.write(json.dumps(item) + "\n")

    return dataset_name

path = create_dataset("https://www.youtube.com/@Fireship", 5)
dataset = datasets.load_dataset("imagefolder", data_dir=path)

print(dataset["train"][0]["text"])