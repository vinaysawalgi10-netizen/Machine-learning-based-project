import streamlit as st
import cohere

# --- Streamlit App Setup ---
st.set_page_config(page_title="Comedy Script Generator", layout="centered")
st.title("🎭 Multilingual Comedy Script Generator")
st.markdown("Create hilarious scripts in your favorite language — powered by **Cohere LLM**.")

# --- Cohere API Setup ---
COHERE_API_KEY = "ln5Bxi7wghNSIGTvLzFBvsiEFDqCdaZ1Z4Cp6ylk"  # Replace with your real key
co = cohere.Client(COHERE_API_KEY)

# --- Supported Languages (Common Global Languages) ---
LANGUAGES = sorted([
    "English", "Hindi", "Spanish", "French", "German", "Mandarin Chinese", "Arabic", "Portuguese",
    "Russian", "Japanese", "Bengali", "Korean", "Tamil", "Telugu", "Marathi", "Turkish", "Urdu",
    "Vietnamese", "Italian", "Malayalam", "Thai", "Gujarati", "Polish", "Dutch", "Persian", "Romanian",
    "Ukrainian", "Hebrew", "Swedish", "Norwegian", "Finnish", "Czech", "Hungarian", "Greek",
    "Kannada", "Punjabi", "Malay", "Indonesian", "Zulu", "Swahili", "Nepali", "Tagalog", "Lao",
    "Burmese", "Sinhala", "Mongolian", "Somali", "Hausa", "Yoruba", "Igbo", "Afrikaans", "Tigrinya"
])

# --- User Inputs ---
language = st.selectbox("🌐 Choose Script Language", LANGUAGES)
genre = st.selectbox("🎭 Choose Comedy Type", ["Stand-up", "Sitcom", "Sketch", "Parody", "Roast"])
topic = st.text_input("📝 Enter a Topic (e.g., Exams, AI, Dating)")
characters = st.text_input("🎭 Optional Characters (comma-separated)")

# --- Generate Button ---
if st.button("🎬 Generate Comedy Script"):
    if not topic:
        st.warning("Please enter a topic to generate a script.")
    else:
        with st.spinner("Writing your script... 🎤"):
            # Build prompt
            prompt = f"""
Write a {genre.lower()} comedy script in {language}.
Topic: {topic}.
{f"Characters involved: {characters}." if characters else ""}
Make it funny, engaging, and culturally appropriate.
Don't limit yourself — be as creative and long as needed.
"""

            try:
                response = co.generate(
                    model="command-r-plus",
                    prompt=prompt,
                    temperature=0.9,
                )
                script = response.generations[0].text.strip()
                st.markdown("### 😂 Your Comedy Script")
                st.text_area("Comedy Script", script, height=400)

            except Exception as e:
                st.error(f"❌ Error: {e}")
