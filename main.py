import streamlit as st
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from ImportScript import *

st.set_page_config(
    page_title="Generative AI Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_and_create_models():
    loaded_story_generator, pipe = load_models()
    return loaded_story_generator, pipe

loaded_story_generator, pipe = load_and_create_models()

def generate_story(prompt, max_length):
    generated_story = loaded_story_generator(prompt, max_length=max_length, do_sample=True,
               repetition_penalty=1.2, temperature=0.7,
               top_p=0.95, top_k=10)
    return generated_story[0]["generated_text"]

def generate_images(prompt, pipe):
    images = pipe(prompt, height=512, width=512,
                num_images_per_prompt=3,
                num_inference_steps=150, guidance_scale=10,
                cross_attention_kwargs={"scale": 0.7})

    return images[0][0], images[0][1], images[0][2]

# Main Streamlit app code
def main():
    # Custom CSS style for the top section with grey background
    top_section_style = """
        <style>
        .top-section {
            padding: 20px;
            color: white;
        }
        .top-section h1 {
            color: #f07807;
        }
        .top-section strong {
            color: #f07807;
        }
        </style>
        """
    button_style = """
        <style>
        .stButton>button {
            width: 100%;
            box-sizing: border-box;
            color: #f07807;
            height: 80px;
            border: 2px solid #f07807;
        }
        </style>
        """

    generated_text_style = """
        <style>
        .generated-story {
            border: 2px solid #f07807; 
            padding: 10px;
            font-size: 25px;
            font-family: 'Arial', sans-serif;
            font-style: italic;
            color: White; 
        }
        </style>
        """

    generated_image_style = """
        <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
        </style>
       """

    subheader_style = """
        <style>
        .custom-subheader {
            color: #f07807;
        }
        </style>
        """
    
    st.markdown(top_section_style, unsafe_allow_html=True)
    st.markdown(button_style, unsafe_allow_html=True)
    st.markdown(generated_text_style, unsafe_allow_html=True)
    st.markdown(generated_image_style, unsafe_allow_html=True)
    st.markdown(subheader_style, unsafe_allow_html=True)

    st.markdown("<div class='top-section'><h1 style='text-align: center; color: #f07807;'>AI Storyteller: Text-to-Story and Image Generation ✨</h1><p style='text-align: center;'>Created by <strong>Ahmed Hany Hereiz</strong><br><br></p><p style='text-align: left;'>Unleash the power of <strong>Artificial Intelligence</strong> to create compelling stories and vivid visuals with our cutting-edge text-to-story and image generation platform. Let your creativity run wild with endless storytelling possibilities!</p></div>", unsafe_allow_html=True)

    # Choose Generation Type
    
    st.markdown("<h3 class='custom-subheader'>Empower Creativity with AI:</h3>", unsafe_allow_html=True)

    options = ['Select','Generate new images from text', 'Generate new story']
    selected_option = st.selectbox('Select Option', options)
    # Model Selection

    if selected_option == 'Select':
        st.write('')

    if selected_option == 'Generate new images from text':

        # Initialize session_state variables
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        if 'prompt' not in st.session_state:
            st.session_state.prompt = ''

        prompt = st.text_area('Enter your prompt to generate image:', st.session_state.prompt, height=100)
            
        # Custom button to handle click and update session_state
        if st.button('Generate New Image using AI', key='generate_button'):
            st.session_state.button_clicked = True
        
        st.write("")
        st.write("")
        st.write("")
        if st.session_state.button_clicked:
            if prompt:
                generated_image1, generated_image2, generated_image3 = generate_images(prompt, pipe)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(
                        generated_image1,
                        width=512,
                        caption='Generated Image 1',
                        use_column_width=False
                    )
                with col2:
                    st.image(
                        generated_image2,
                        width=512,
                        caption='Generated Image 2',
                        use_column_width=False
                    )
                with col3:
                    st.image(
                        generated_image3,
                        width=512,
                        caption='Generated Image 3',
                        use_column_width=False
                    )

        # Reset button_clicked flag to False
        st.session_state.button_clicked = False

    if selected_option == 'Generate new story':
        
        # Initialize session_state variables
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        if 'prompt' not in st.session_state:
            st.session_state.prompt = ''

        max_length = st.slider('Select maximum number of generated words in story:', 100, 1000, 200, step=50)
        prompt = st.text_area('Enter your prompt for the story:', st.session_state.prompt, height=100)
        
        # Custom button to handle click and update session_state
        button_labels = ["Generate Superhero story with AI",
                         "Generate Science Fiction story with AI",
                         "Generate Horror story with AI",
                         "Generate Thriller story with AI",
                         "Generate Action story with AI",
                         "Generate Drama story with AI"]

        with st.container():
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if col.button(button_labels[j]):
                    st.session_state.button_clicked = True

                    if j == 0:
                        input_prompt = "<BOS> <superhero> " + prompt
                        generated_story = generate_story(input_prompt, max_length)[17:]
                        story_type = "Superhero"
                    elif j == 1:
                        input_prompt = "<BOS> <sci_fi> " + prompt
                        generated_story = generate_story(input_prompt, max_length)[14:]
                        story_type = "Science Fiction"
                    elif j == 2:
                        input_prompt = "<BOS> <horror> " + prompt
                        generated_story = generate_story(input_prompt, max_length)[14:]
                        story_type = "Horror"

        with st.container():
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = j + 3  
                if idx < len(button_labels):
                    if col.button(button_labels[idx]):
                        st.session_state.button_clicked = True

                        if idx == 3:
                            input_prompt = "<BOS> <thriller> " + prompt
                            generated_story = generate_story(input_prompt, max_length)[16:]
                            story_type = "Thriller"
                        elif idx == 4:
                            input_prompt = "<BOS> <action> " + prompt
                            generated_story = generate_story(input_prompt, max_length)[14:]
                            story_type = "Action"
                        elif idx == 5:
                            input_prompt = "<BOS> <drama> " + prompt
                            generated_story = generate_story(input_prompt, max_length)[13:]
                            story_type = "Drama"

        # Display the generated story
        if st.session_state.button_clicked:
            st.markdown(f"<h3 class='custom-subheader'>Generated {story_type} story:</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='generated-story'>{generated_story}</div>", unsafe_allow_html=True)


        # Reset button_clicked flag to False
        st.session_state.button_clicked = False

if __name__ == "__main__":
    main()
