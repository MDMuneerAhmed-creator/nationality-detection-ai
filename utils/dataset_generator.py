"""
Dataset Structure & Synthetic Image Generator for Scratch-Training 4 Custom CNN Models.

Creates datasets directory hierarchy:
- datasets/nationality/{Indian, American, African, Japanese, Chinese, German, French}
- datasets/emotion/{Happy, Sad, Angry, Fear, Neutral, Disgust, Surprise}
- datasets/age/{0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 60+}
- datasets/dress_color/{Black, Blue, White, Green, Yellow, Red, Orange, Pink, Purple, Brown, Grey}
"""

import os
import numpy as np
from PIL import Image, ImageDraw

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "samples")

NATIONALITIES = ["Indian", "American", "African", "Japanese", "Chinese", "German", "French"]
EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Neutral", "Disgust", "Surprise"]
AGE_GROUPS = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]
DRESS_COLORS = ["Black", "Blue", "White", "Green", "Yellow", "Red", "Orange", "Pink", "Purple", "Brown", "Grey"]

# Color mapping for dress colors
COLOR_RGB_MAP = {
    "Black": (30, 30, 30),
    "Blue": (40, 90, 220),
    "White": (240, 240, 240),
    "Green": (40, 180, 80),
    "Yellow": (240, 220, 40),
    "Red": (220, 40, 40),
    "Orange": (240, 130, 30),
    "Pink": (240, 110, 170),
    "Purple": (140, 50, 200),
    "Brown": (130, 70, 30),
    "Grey": (120, 120, 120)
}

# Skin tone palettes for synthetic representations
SKIN_TONES = {
    "Indian": [(198, 134, 66), (160, 100, 45), (210, 150, 90)],
    "American": [(240, 195, 150), (220, 160, 120), (245, 215, 180)],
    "African": [(90, 55, 30), (60, 35, 18), (115, 75, 45)],
    "Japanese": [(245, 215, 185), (235, 200, 170), (250, 225, 195)],
    "Chinese": [(240, 210, 175), (230, 195, 160), (245, 220, 185)],
    "German": [(250, 220, 190), (240, 205, 175), (255, 230, 200)],
    "French": [(245, 210, 180), (235, 195, 165), (248, 220, 190)]
}

def generate_synthetic_portrait(
    nationality: str = "Indian",
    emotion: str = "Happy",
    age: str = "21-30",
    dress_color: str = "Red",
    width: int = 128,
    height: int = 128
) -> Image.Image:
    """Generates a clean synthetic portrait image with distinct visual signals for training/testing."""
    img = Image.new("RGB", (width, height), (240, 242, 245))
    draw = ImageDraw.Draw(img)

    # 1. Background & Dress (bottom 45% of image)
    d_color = COLOR_RGB_MAP.get(dress_color, (100, 100, 100))
    draw.rectangle([0, int(height * 0.55), width, height], fill=d_color)

    # 2. Neck
    tones = SKIN_TONES.get(nationality, [(200, 150, 100)])
    skin_color = tones[np.random.choice(len(tones))]
    draw.rectangle([int(width * 0.4), int(height * 0.5), int(width * 0.6), int(height * 0.65)], fill=skin_color)

    # 3. Head (Oval in upper region)
    head_left, head_top = int(width * 0.25), int(height * 0.15)
    head_right, head_bottom = int(width * 0.75), int(height * 0.60)
    draw.ellipse([head_left, head_top, head_right, head_bottom], fill=skin_color, outline=(40, 40, 40), width=1)

    # 4. Eyes
    eye_y = int(height * 0.32)
    left_eye_x = int(width * 0.38)
    right_eye_x = int(width * 0.62)
    eye_r = 3
    draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))
    draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))

    # 5. Mouth expression
    mouth_y = int(height * 0.48)
    if emotion == "Happy":
        draw.arc([int(width * 0.40), mouth_y - 4, int(width * 0.60), mouth_y + 8], start=0, end=180, fill=(180, 40, 40), width=2)
    elif emotion == "Sad":
        draw.arc([int(width * 0.40), mouth_y, int(width * 0.60), mouth_y + 12], start=180, end=360, fill=(180, 40, 40), width=2)
    elif emotion == "Angry":
        draw.line([int(width * 0.42), mouth_y, int(width * 0.58), mouth_y], fill=(180, 30, 30), width=3)
        draw.line([int(width * 0.35), eye_y - 6, int(width * 0.44), eye_y - 2], fill=(30, 30, 30), width=2)
        draw.line([int(width * 0.65), eye_y - 6, int(width * 0.56), eye_y - 2], fill=(30, 30, 30), width=2)
    elif emotion == "Surprise":
        draw.ellipse([int(width * 0.45), mouth_y - 3, int(width * 0.55), mouth_y + 7], fill=(180, 40, 40))
    else: # Neutral, Fear, Disgust
        draw.line([int(width * 0.42), mouth_y, int(width * 0.58), mouth_y], fill=(100, 30, 30), width=2)

    # 6. Age indicators (wrinkle lines or hair color)
    hair_color = (40, 30, 20)
    if age in ["51-60", "60+"]:
        hair_color = (200, 200, 205)
        # Add subtle forehead lines
        draw.line([int(width * 0.35), int(height * 0.22), int(width * 0.65), int(height * 0.22)], fill=(120, 90, 70), width=1)
        draw.line([int(width * 0.38), int(height * 0.25), int(width * 0.62), int(height * 0.25)], fill=(120, 90, 70), width=1)
    
    # Hair patch
    draw.chord([head_left - 2, head_top - 5, head_right + 2, head_top + 25], start=180, end=360, fill=hair_color)

    return img

def create_full_datasets(images_per_class: int = 15):
    """Populates datasets/ folder structure with synthetic samples for scratch training."""
    print("Generating dataset folder structure and synthetic image samples...")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

    # 1. Nationality Dataset
    for nat in NATIONALITIES:
        folder = os.path.join(DATASETS_DIR, "nationality", nat)
        os.makedirs(folder, exist_ok=True)
        for i in range(images_per_class):
            img = generate_synthetic_portrait(
                nationality=nat,
                emotion=np.random.choice(EMOTIONS),
                age=np.random.choice(AGE_GROUPS),
                dress_color=np.random.choice(DRESS_COLORS)
            )
            img.save(os.path.join(folder, f"{nat.lower()}_{i+1}.jpg"))

    # 2. Emotion Dataset
    for emo in EMOTIONS:
        folder = os.path.join(DATASETS_DIR, "emotion", emo)
        os.makedirs(folder, exist_ok=True)
        for i in range(images_per_class):
            img = generate_synthetic_portrait(
                nationality=np.random.choice(NATIONALITIES),
                emotion=emo,
                age=np.random.choice(AGE_GROUPS),
                dress_color=np.random.choice(DRESS_COLORS)
            )
            img.save(os.path.join(folder, f"{emo.lower()}_{i+1}.jpg"))

    # 3. Age Dataset
    for age in AGE_GROUPS:
        folder = os.path.join(DATASETS_DIR, "age", age)
        os.makedirs(folder, exist_ok=True)
        for i in range(images_per_class):
            img = generate_synthetic_portrait(
                nationality=np.random.choice(NATIONALITIES),
                emotion=np.random.choice(EMOTIONS),
                age=age,
                dress_color=np.random.choice(DRESS_COLORS)
            )
            clean_age = age.replace("+", "_plus")
            img.save(os.path.join(folder, f"age_{clean_age}_{i+1}.jpg"))

    # 4. Dress Color Dataset
    for dc in DRESS_COLORS:
        folder = os.path.join(DATASETS_DIR, "dress_color", dc)
        os.makedirs(folder, exist_ok=True)
        for i in range(images_per_class):
            img = generate_synthetic_portrait(
                nationality=np.random.choice(NATIONALITIES),
                emotion=np.random.choice(EMOTIONS),
                age=np.random.choice(AGE_GROUPS),
                dress_color=dc
            )
            img.save(os.path.join(folder, f"dress_{dc.lower()}_{i+1}.jpg"))

    # 5. Generate preset sample images for GUI instant prediction testing
    test_samples = [
        ("indian_happy_red.jpg", "Indian", "Happy", "21-30", "Red"),
        ("american_neutral_blue.jpg", "American", "Neutral", "41-50", "Blue"),
        ("african_surprise_yellow.jpg", "African", "Surprise", "11-20", "Yellow"),
        ("japanese_sad_black.jpg", "Japanese", "Sad", "31-40", "Black"),
        ("german_angry_green.jpg", "German", "Angry", "51-60", "Green")
    ]

    for filename, nat, emo, age, d_col in test_samples:
        img = generate_synthetic_portrait(nationality=nat, emotion=emo, age=age, dress_color=d_col, width=256, height=256)
        img.save(os.path.join(SAMPLE_IMAGES_DIR, filename))

    print("Datasets structure and sample images generated successfully.")

if __name__ == "__main__":
    create_full_datasets(images_per_class=20)
