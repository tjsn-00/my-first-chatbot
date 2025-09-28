"""
Streamlit app: Gemini-2.5-flash travel planner chatbot
Features:
- Uses google-genai python client to call gemini-2.5-flash
- Reads GEMINI_API_KEY from Streamlit secrets (st.secrets["GEMINI_API_KEY"]) and sets it into the environment
- Conversation UI: input box + history display
- Asks travel questions (budget, travel style, duration, people) and generates an itinerary
- Attempts streaming responses; falls back to non-streaming if streaming isn't available

How to run:
1) pip install streamlit google-genai
2) Create a Streamlit secrets.toml file (see below) or set your secrets in Streamlit Cloud.

# .streamlit/secrets.toml example:
# [GEMINI]
# GEMINI_API_KEY = "your_api_key_here"
# Note: In this app we expect st.secrets["GEMINI_API_KEY"] to exist. If you use Streamlit Cloud, set Secrets there.

Run:
    streamlit run streamlit_gemini_travel_chatbot.py

"""

import os
import streamlit as st
from google import genai
from google.genai import types
import time

# ----------------------
# App configuration
# ----------------------
st.set_page_config(page_title="Travel Planner — Gemini", layout="wide")

# Ensure user is explicit about secrets usage
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Add it to .streamlit/secrets.toml as GEMINI_API_KEY = \"...\"")
    st.stop()

# Put the key into environment so google-genai client picks it up
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Create GenAI client (will use GEMINI_API_KEY from env)
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

# ----------------------
# Helpers
# ----------------------

def ensure_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, text)
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}


def add_message(role, text):
    st.session_state.history.append((role, text))


# Build a system prompt that instructs the model to act as a travel planner
SYSTEM_PROMPT = (
    "You are a friendly, precise travel-planning assistant. "
    "Ask any clarification questions required to produce a high-quality multi-day travel itinerary, "
    "then produce: 1) destination summary, 2) recommended accommodation (3 options), 3) day-by-day itinerary with timed suggestions for places to visit and meals, 4) estimated budget breakdown. "
    "Be concise and present the final itinerary as numbered days with bullet points."
)

# ----------------------
# UI layout
# ----------------------
ensure_session()

left, right = st.columns([2, 3])

with left:
    st.header("Travel Planner Chatbot")

    # Quick profile fill form
    with st.form("profile_form"):
        st.subheader("Trip details (fill / update)")
        destination = st.text_input("Preferred destination (optional)", value=st.session_state.user_profile.get("destination", ""))
        start_date = st.date_input("Start date (optional)", value=st.session_state.user_profile.get("start_date", None))
        nights = st.number_input("Number of nights", min_value=0, max_value=30, value=st.session_state.user_profile.get("nights", 2))
        people = st.number_input("Number of people", min_value=1, max_value=20, value=st.session_state.user_profile.get("people", 1))
        budget = st.text_input("Budget (per person or total, e.g. '₩300,000 per person')", value=st.session_state.user_profile.get("budget", ""))
        travel_style = st.selectbox("Travel style", options=["Balanced", "Relaxed/slow", "Active/adventure", "Food-focused", "Luxury", "Budget"], index=0)
        submit_profile = st.form_submit_button("Save trip details")

    if submit_profile:
        st.session_state.user_profile = {
            "destination": destination,
            "start_date": str(start_date) if start_date else "",
            "nights": int(nights),
            "people": int(people),
            "budget": budget,
            "travel_style": travel_style,
        }
        st.success("Trip details saved.")

    st.markdown("---")

    # User input box for chat
    user_input = st.text_area("Message to assistant", value="", placeholder="Ask for a full itinerary, or say 'plan my trip' to start.")
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please write a message before sending.")
        else:
            add_message("user", user_input.strip())

    st.markdown("\n---\nQuick actions:")
    col1, col2 = st.columns(2)
    if col1.button("Plan full itinerary now"):
        # Compose a prompt that instructs the model to use saved profile data
        profile = st.session_state.user_profile
        prompt_parts = ["Please create a travel plan."]
        if profile.get("destination"):
            prompt_parts.append(f"Destination: {profile['destination']}")
        if profile.get("nights") is not None:
            prompt_parts.append(f"Nights: {profile['nights']}")
        if profile.get("people"):
            prompt_parts.append(f"People: {profile['people']}")
        if profile.get("budget"):
            prompt_parts.append(f"Budget: {profile['budget']}")
        if profile.get("travel_style"):
            prompt_parts.append(f"Travel style: {profile['travel_style']}")
        prompt_parts.append("If any information is missing, ask a short clarifying question before producing the full itinerary.")
        composed = "\n".join(prompt_parts)
        add_message("user", composed)

    if col2.button("Clear chat"):
        st.session_state.history = []
        st.success("Chat cleared.")

with right:
    st.subheader("Conversation")
    box = st.container()

    # Display history
    with box:
        for role, text in st.session_state.history:
            if role == "system":
                st.info(f"SYSTEM: {text}")
            elif role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")

# ----------------------
# If there is a new user message in the last item, call the model
# ----------------------
if len(st.session_state.history) > 0 and st.session_state.history[-1][0] == "user":
    latest_user = st.session_state.history[-1][1]

    # Build conversation formatted for Gemini (we'll send system then user)
    contents = [
        {"parts": [{"text": SYSTEM_PROMPT}]},
        {"parts": [{"text": latest_user}]},
    ]

    # Insert a placeholder assistant message to be replaced by streaming content
    add_message("assistant", "")

    # Stream the response if possible
    assistant_text = ""
    streamed = False
    stream_placeholder = st.empty()

    try:
        # Attempt streaming API
        stream = None
        # The google-genai library naming for streaming may vary across versions; try common names in order.
        if hasattr(client.models, "generate_content_stream"):
            stream = client.models.generate_content_stream(model=MODEL_NAME, contents=contents)
        elif hasattr(client.models, "generate_content_streaming"):
            stream = client.models.generate_content_streaming(model=MODEL_NAME, contents=contents)
        elif hasattr(client.models, "stream_generate_content"):
            stream = client.models.stream_generate_content(model=MODEL_NAME, contents=contents)
        elif hasattr(client.models, "stream_generate"):
            stream = client.models.stream_generate(model=MODEL_NAME, contents=contents)

        if stream is not None:
            streamed = True
            for chunk in stream:
                # depending on client, chunk may be a dict or object with .text
                if hasattr(chunk, "text"):
                    piece = chunk.text
                elif isinstance(chunk, dict) and "text" in chunk:
                    piece = chunk["text"]
                else:
                    piece = str(chunk)
                assistant_text += piece
                # update live UI (replace the placeholder assistant message)
                st.session_state.history[-1] = ("assistant", assistant_text)
                stream_placeholder.markdown(f"**Assistant (streaming):** {assistant_text}")
            # after streaming completes, no-op
    except Exception as e:
        # Streaming not available or failed; we'll fall back to non-streaming below
        st.warning(f"Streaming failed or not supported in this environment: {e}")
        streamed = False

    if not streamed:
        try:
            # Non-streaming call
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
            # resp.text is the convenient helper from quickstart
            text = getattr(resp, "text", None) or str(resp)
            assistant_text = text
            st.session_state.history[-1] = ("assistant", assistant_text)
        except Exception as e:
            # Provide user-friendly error
            err_msg = f"Error calling Gemini API: {e}. Check your API key and network."
            st.session_state.history[-1] = ("assistant", err_msg)

    # final rendering
    st.experimental_rerun()

# ----------------------
# End of file
# ----------------------
