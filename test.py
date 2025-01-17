from dotenv import load_dotenv
import os
import spotipy
import time
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


os.system("start spotify")
time.sleep(3)

redirect_uri = 'http://localhost:8888/callback'  # Make sure this matches your Spotify app settings

# Set up the Spotify OAuth object with user authentication
scope = 'user-modify-playback-state user-read-playback-state'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))


def play_song(song_name):
    # Search for the song
    results = sp.search(q=song_name, type='track', limit=1)
    if results['tracks']['items']:
        track_uri = results['tracks']['items'][0]['uri']
        print(f"Playing: {results['tracks']['items'][0]['name']}")

        # Check if there are available devices
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']  # Use the first available device
            # Start playback with the device ID
            sp.start_playback(device_id=device_id, uris=[track_uri])
        else:
            print("No active devices found. Please open Spotify on a device.")
    else:
        print("Song not found.")

# Call the function with the song name
# play_song("Shameless - Camila Cabello")
