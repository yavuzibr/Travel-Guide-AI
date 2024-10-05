import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from transformers import pipeline

# API Configuration
apikey=os.environ.get("GROQ_API_KEY")
client = Groq(api_key=apikey)

llama31_model = 'llama-3.1-70b-versatile'

system_prompt = """
You are a tourist guide give useful and historical information about the places that user gives to you
"""

# Load the model
model = YOLO("best.pt")
result = None

# Ask the user to upload an image
uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Resim', use_column_width=True)

    # Create a temporary file and save the image to that file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

        # Predict the image using the model
        result = model.predict(source=temp_file_path)

    # Extract class information from the result
    class_label = None
    for r in result:
        if r.boxes is not None:
            class_indices = r.boxes.cls.cpu().numpy()  # Get the class indices
            if len(class_indices) > 0:
                class_label = r.names[int(class_indices[0])]  # Get the first class label

    # If class label is found, proceed with the chatbot
    if class_label:
        user_prompt = f"Give information about {class_label}"

        def prompt_guesser(client, user_prompt):
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=llama31_model
            )
            return chat_completion.choices[0].message.content

        st.write(class_label)
        st.write(prompt_guesser(client, user_prompt))
    else:
        st.write("No object detected in the image.")
