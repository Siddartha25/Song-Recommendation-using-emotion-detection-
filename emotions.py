import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from test import play_song

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


mood_songs = {
    "Happy": [
        "Walking on Sunshine", "Happy", "Can't Stop the Feeling!", "Good as Hell", "Uptown Funk",
        "Dancing Queen", "Good Vibrations", "I Got You (I Feel Good)", "Lovely Day", "Shake It Off",
        "Roar", "September", "All Star", "I'm a Believer", "Best Day of My Life",
        "Sunflower", "Sugar", "Counting Stars", "On Top of the World", "A Sky Full of Stars",
        "One Kiss", "Wake Me Up Before You Go-Go", "Sweet Caroline", "I Gotta Feeling", "Party Rock Anthem",
        "Hey Ya!", "Celebration", "Get Lucky", "What a Wonderful World", "Don't Worry, Be Happy",
        "Mr. Blue Sky", "Freedom", "Raise Your Glass", "Good Life", "You Make My Dreams", 
        "Happy Together", "Shake Your Groove Thing", "Can't Stop", "Dog Days Are Over", "Smile",
        "American Boy", "Cheerleader", "Moves Like Jagger", "Pocketful of Sunshine", "Ain't No Mountain High Enough",
        "Some Nights", "Electric Feel", "Fireflies", "Let's Go Crazy", "Three Little Birds"
    ],
    "Sad": [
        "Gypsy Woman", "Underdxse", "Get Back", "Little Things", "Archangel",
        "Someone Like You", "Let Her Go", "Skinny Love", "When I Was Your Man", "The Night We Met",
        "Tears Dry on Their Own", "Fast Car", "Hurt", "Everybody Hurts", "Fix You",
        "Someone You Loved", "Goodbye My Lover", "Dancing On My Own", "Stay With Me", "All I Want",
        "The A Team", "I Will Remember You", "Breathe Me", "Nothing Compares 2 U", "The Scientist",
        "I Can't Make You Love Me", "Unbreak My Heart", "Let It Go", "In the Arms of an Angel", "Say Something",
        "Yesterday", "With or Without You", "Runaway", "Landslide", "Shallow",
        "Creep", "This Woman's Work", "Mad World", "Under the Bridge", "Everybody's Changing",
        "Chasing Cars", "Gravity", "Wish You Were Here", "Summertime Sadness", "Falling",
        "Jealous", "I'm Not the Only One", "The Sound of Silence", "Back to Black", "Lost Without You"
    ],
    "Surprised": [
        "Don't Stop Me Now", "Firework", "Thunderstruck", "Born to Run", "We Will Rock You",
        "Eye of the Tiger", "Survivor", "Come As You Are", "Here Comes the Sun", "I Want to Break Free",
        "Shout", "Happy", "Pump It", "We Are the Champions", "Simply the Best",
        "It's My Life", "Viva la Vida", "Can't Hold Us", "Under Pressure", "Baba O'Riley",
        "Living on a Prayer", "Stronger", "Runnin'", "Radioactive", "Counting Stars",
        "Sweet Child O' Mine", "Another One Bites the Dust", "Paint It Black", "Fortunate Son", "Don't Stop Believin'",
        "Welcome to the Jungle", "Smells Like Teen Spirit", "Paradise City", "Dream On", "Highway to Hell",
        "Seven Nation Army", "Uprising", "I Love Rock 'n' Roll", "Born to Be Wild", "Whole Lotta Love",
        "Break Free", "Burn", "Shake It Out", "Electric Feel", "Ghostbusters", 
        "Shut Up and Dance", "Do You Believe in Magic", "Rolling in the Deep", "Final Countdown", "Bohemian Rhapsody"
    ],
    "Fearful": [
        "In the House - In a Heartbeat", "The Fog", "The Host of Seraphim", "Lux Aeterna", "The Killing Moon",
        "Ghosts", "Tubular Bells", "Thriller", "Don't Fear the Reaper", "Psycho",
        "Red Right Hand", "In the Air Tonight", "Highway to Hell", "Somebody's Watching Me", "Hells Bells",
        "Bury a Friend", "Hurt", "Scary Monsters (and Super Creeps)", "Haunted", "Disturbia",
        "Close to Me", "Wicked Game", "Riders on the Storm", "People Are Strange", "Suspicious Minds",
        "Evil Ways", "Toxic", "I Put a Spell on You", "I Am the Walrus", "Crazy Train",
        "Time Is on My Side", "Welcome to My Nightmare", "In the End", "House of the Rising Sun", "Bad Guy",
        "The Sound of Silence", "Personal Jesus", "The Less I Know the Better", "Take Me to Church", "Glory Box",
        "No One Knows", "Space Oddity", "Knights of Cydonia", "Black Magic Woman", "Down by the Water",
        "How Soon Is Now?", "Clint Eastwood", "The Passenger", "Fade to Black", "Dead Souls"
    ],
    "Neutral": [
        "Clocks", "Viva La Vida", "Waiting on the World to Change", "Hello", "Midnight City",
        "Imagine", "Somewhere Only We Know", "Wonderwall", "Fast Car", "Let It Be",
        "House of the Rising Sun", "Bittersweet Symphony", "Stayin' Alive", "Come Together", "Let’s Stay Together",
        "Sunday Morning", "Dreams", "Norwegian Wood", "California Dreamin'", "Both Sides Now",
        "Take It Easy", "Easy", "More Than a Feeling", "Wild World", "No Woman, No Cry",
        "Shallow", "Landslide", "Hotel California", "Africa", "Yesterday",
        "Come As You Are", "Ramble On", "Blackbird", "New York State of Mind", "Scar Tissue",
        "Every Breath You Take", "Hey Jude", "How Deep Is Your Love", "You've Got a Friend", "Rocket Man",
        "After the Gold Rush", "If You Leave Me Now", "Angie", "The Joker", "The Boxer",
        "Turn the Page", "Don't Let the Sun Go Down on Me", "I Am a Rock", "Wonderful Tonight", "Wild Horses"
    ],
    "Angry": [
        "Killing in the Name", "Break Stuff", "Bodies", "You Oughta Know", "Head Like a Hole",
        "Du Hast", "Enter Sandman", "Sympathy for the Devil", "Stricken", "We Will Rock You",
        "I'm Not Okay (I Promise)", "Last Resort", "Bulls on Parade", "Down with the Sickness", "Walk",
        "Back in Black", "Before I Forget", "Psychosocial", "The Way I Am", "Numb",
        "Bring Me to Life", "Freak on a Leash", "Unforgiven", "Given Up", "Hard to Handle",
        "Kryptonite", "Can't Hold Us", "Stronger", "It's My Life", "Paranoid",
        "Livin' la Vida Loca", "I Will Survive", "The Pretender", "Seven Nation Army", "My Way",
        "Highway to Hell", "Whole Lotta Love", "Money for Nothing", "Bad Reputation", "American Idiot",
        "Thunderstruck", "Baba O'Riley", "Born to Run", "I Hate Everything About You", "Get Up, Stand Up",
        "Time Is on My Side", "Faint", "Break", "Iron Man", "Crawling"
    ],
    "Disgusted": [
        "Break Free", "Look What You Made Me Do", "U + Ur Hand", "Toxic", "I Hate Everything About You",
        "You Oughta Know", "Bad Guy", "F***in' Perfect", "So What", "Fighter",
        "The Way I Am", "Can't Be Tamed", "IDGAF", "Don't Cha", "Hard Out Here",
        "Bury a Friend", "Gives You Hell", "Control", "Flesh Without Blood", "Everything Zen",
        "My Own Worst Enemy", "Unpretty", "Take a Bow", "Sorry Not Sorry", "Go to Hell",
        "Blank Space", "Misery Business", "Hotline Bling", "Bad Blood", "Wrecking Ball",
        "Oops!... I Did It Again", "Cry Me a River", "Roxanne", "Complicated", "Stressed Out",
        "I Knew You Were Trouble", "Maneater", "Take a Bow", "Sweet but Psycho", "Tears Dry on Their Own",
        "Jar of Hearts", "Truth Hurts", "Back to Black", "Don't Speak", "Rolling in the Deep",
        "Ex's & Oh's", "Sorry", "Shout", "Torn", "I'm So Sick"
    ]
}

# Function to print detected mood and random songs
def print_mood_and_songs(detected_mood):
    if detected_mood in mood_songs:
        print(f"Detected mood is: {detected_mood}")
        print("Random songs based on detected mood:")
        # Randomly shuffle the song list and pick 1 song
        songs = random.sample(mood_songs[detected_mood], 1)
        for song in songs:
            return song
            # print(f"- {song}")
    else:
        print("Mood not recognized.")

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')



# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    last_capture_time = time.time()
    
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        current_time = time.time()
        if current_time - last_capture_time >= 0.1:
        # Update the last capture time
            last_capture_time = current_time
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame,print_mood_and_songs( emotion_dict[maxindex]) , (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                play_song(print_mood_and_songs( emotion_dict[maxindex]))
                print("Emotion-",emotion_dict[maxindex],"/nSong-",print_mood_and_songs( emotion_dict[maxindex]))

            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()