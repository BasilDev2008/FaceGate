import cv2  # for image processing
import numpy as np  # for math
import os  # for file management
import librosa  # for audio processing
def augment_faces(person_name):
    input_folder = f"../data/faces/{person_name}" # folder containing all photos
    output_folder = f"../data/faces/{person_name}_augmented" # where the augmented photos are
    os.makedirs(output_folder, exist_ok=True)
    images = os.listdir(input_folder)
    count = 0 # track num of augmented photos
    print(f"Augmenting face data for {person_name}")
    for image_file in images:
        img = cv2.imread(f"{input_folder}/{image_file}")
        if img is None:
            continue
        cv2.imwrite(f"{output_folder}/{count}.jpg", img)
        count+=1
        #first augmentation - flip photo horizontally
        flipped = cv2.flip(img, 1)
        cv2.imwrite(f"{output_folder}/{count}.jpg", flipped)
        count+=1
        #second augmentation rotate left
        h, w = img.shape[:2] # dimensions of image
        matrix = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0) # rotate by 10 degrees
        rotated_left = cv2.warpAffine(img, matrix, (w, h))
        cv2.imwrite(f"{output_folder}/{count}.jpg", rotated_left)
        count+=1
        # [[cos(10°),  sin(10°),  tx],
        #[-sin(10°), cos(10°),  ty]]
        # augmentation 3 — rotate slightly right
        matrix = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)  # rotate -10 degrees
        rotated_right = cv2.warpAffine(img, matrix, (w, h))  # apply rotation
        cv2.imwrite(f"{output_folder}/{count}.jpg", rotated_right)
        count += 1
        # augmentation 4 — brighten
        brightened = cv2.convertScaleAbs(img, alpha=1.3, beta=30)  # increase brightness
        cv2.imwrite(f"{output_folder}/{count}.jpg", brightened)
        count += 1
        # augmentation 5 — darken
        darkened = cv2.convertScaleAbs(img, alpha=0.7, beta=-30)  # decrease brightness
        cv2.imwrite(f"{output_folder}/{count}.jpg", darkened)
        count += 1
        # augmentation 6 — add blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)  # apply gaussian blur
        cv2.imwrite(f"{output_folder}/{count}.jpg", blurred)
        count += 1
        # augmentation 7 — add noise
        noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)  # generate random noise
        noisy = cv2.add(img, noise)  # add noise to image
        cv2.imwrite(f"{output_folder}/{count}.jpg", noisy)
        count += 1
        print(f"Face augmentation complete. Created {count} images from {len(images)} originals.")
    def augment_voices(person_name):
        input_folder = f"../data/voices/{person_name}"  # where original recordings are
        output_folder = f"../data/voices/{person_name}_augmented"  # where augmented recordings go
        os.makedirs(output_folder, exist_ok=True)  # create output folder
        recordings = os.listdir(input_folder)
        count = 0
        sample_rate = 16000
        print(f"\nAugmenting voice data for {person_name}...")
        for recording_file in recordings:
            audio = np.load(f"{input_folder}/{recording_file}")
            np.save(f"{output_folder}/{count}.npy", audio)
            count+=1
            # augmentation 1 — add background noise
            noise = np.random.randn(len(audio)) * 0.005  # generate small random noise
            noisy_audio = audio + noise  # add noise to audio
            np.save(f"{output_folder}/{count}.npy", noisy_audio)
            count += 1
            # augmentation 2 — shift pitch up
            pitched_up = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)  # shift pitch up 2 steps
            np.save(f"{output_folder}/{count}.npy", pitched_up)
            count += 1
            # augmentation 3 — shift pitch down
            pitched_down = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2)  # shift pitch down 2 steps
            np.save(f"{output_folder}/{count}.npy", pitched_down)
            count += 1
            # augmentation 4 — speed up slightly
            stretched = librosa.effects.time_stretch(audio, rate=1.1)  # speed up by 10%
            np.save(f"{output_folder}/{count}.npy", stretched)
            count += 1

            # augmentation 5 — slow down slightly
            slowed = librosa.effects.time_stretch(audio, rate=0.9)  # slow down by 10%
            np.save(f"{output_folder}/{count}.npy", slowed)
            count += 1
            print(f"Voice augmentation complete. Created {count} recordings from {len(recordings)} originals.")
        if __name__ == "__main__":
            name = input("Enter person's name: ")  # ask for name
            augment_faces(name)  # augment face data
            augment_voices(name)  # augment voice data


            








