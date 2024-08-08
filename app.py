import streamlit as st 
from lida import Manager, TextGenerationConfig
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
from types import SimpleNamespace
import speech_recognition as sr

# Set up Gemini API with your provided key
genai.configure(api_key="AIzaSyCyDTqVlRyiVkNuM2PhaZJxFuz74acEqzo")

def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Create a custom LLM class for Gemini Flash
class GeminiFlashLLM:
    def __init__(self):
        self.provider = "gemini"
        self.model = "gemini-1.5-flash"
        self._model = genai.GenerativeModel('gemini-1.5-flash')

    def generate(self, messages=None, prompt=None, config=None):
        if messages:
            prompt = " ".join([m["content"] for m in messages])
        
        if prompt is None:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        response = self._model.generate_content(prompt)
        
        return SimpleNamespace(
            text=[{"content": response.text}]
        )

# Initialize LIDA with custom Gemini Flash LLM
lida = Manager(text_gen=GeminiFlashLLM())
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gemini-1.5-flash", use_cache=True)

# Function to perform speech recognition
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return ""

# Initialize session state
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Streamlit app
st.title("LIDA Data Analysis App")

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph"])

if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        
        try:
            summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
            st.write("Summary:", summary)
            
            goals = lida.goals(summary, n=2, textgen_config=textgen_config)
            st.write("Goals:")
            for goal in goals:
                st.write(goal)
            
            i = 0
            library = "seaborn"
            textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
            charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
            if charts:
                img_base64_string = charts[0].raster
                img = base64_to_image(img_base64_string)
                st.image(img)
            else:
                st.warning("No chart could be generated for this summary. This might be due to the nature of the data or the specific goal.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            text_area = st.text_area("Query your Data to Generate Graph", 
                                     value=st.session_state.query_text, 
                                     height=100, 
                                     key="query_input")
        with col2:
            st.write("")
            st.write("")
            if st.button("🎤"):
                speech_text = speech_to_text()
                if speech_text:
                    st.session_state.query_text = speech_text
                    st.experimental_rerun()

        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                try:
                    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                    summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                    user_query = text_area
                    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                    if charts:
                        image_base64 = charts[0].raster
                        img = base64_to_image(image_base64)
                        st.image(img)
                    else:
                        st.warning("No chart could be generated for this query. Please try a different question.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a query before generating a graph.")

st.sidebar.info("This app uses LIDA and Gemini to analyze and visualize your data.")