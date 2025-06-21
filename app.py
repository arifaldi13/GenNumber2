import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np

# --- 1. Define Model Architecture ---
# This MUST be the same architecture as in the training script.

LATENT_DIM = 100
NUM_CLASSES = 10
EMBEDDING_DIM = 10

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, EMBEDDING_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBEDDING_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embeddings = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_embeddings), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# --- 2. Load the Trained Model ---
# Use a cache to load the model only once
@st.cache_resource
def load_model():
    # IMPORTANT: The model was trained on GPU, but we run it on CPU for inference in Streamlit.
    # `map_location=torch.device('cpu')` is crucial for this.
    model = Generator()
    model.load_state_dict(torch.load('cgan_generator.pth', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

generator = load_model()

# --- 3. Create the Streamlit Web App UI ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a Conditional GAN model trained from scratch.")

# --- UI Components ---
st.sidebar.header("Controls")
digit_to_generate = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.sidebar.button("Generate Images"):
    st.subheader(f"Generated images of digit: {digit_to_generate}")

    # Generate 5 images
    num_images = 5
    with torch.no_grad():
        # Prepare noise and labels
        noise = torch.randn(num_images, LATENT_DIM)
        labels = torch.LongTensor([digit_to_generate] * num_images)

        # Generate images
        generated_imgs = generator(noise, labels)

        # Post-process for display: un-normalize from [-1, 1] to [0, 1]
        generated_imgs = generated_imgs * 0.5 + 0.5

        # Use columns to display images side-by-side
        cols = st.columns(num_images)
        for i in range(num_images):
            with cols[i]:
                # Convert tensor to numpy for display
                img_np = generated_imgs[i].squeeze().numpy()
                st.image(img_np, caption=f"Sample {i+1}", use_column_width=True)
else:
    st.info("Select a digit and click 'Generate Images' in the sidebar.")