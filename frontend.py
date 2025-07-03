import streamlit as st
import requests
import os

# Try to get the API base URL from an environment variable, otherwise use a default or ask.
# For the invoke endpoint, it's typically /<graph_id>/invoke
# The graph_id is 'research_agent' from langgraph.json
API_INVOKE_URL_DEFAULT = "" # User will need to provide this if not set via env
API_INVOKE_URL = os.getenv("API_INVOKE_URL", API_INVOKE_URL_DEFAULT)

st.set_page_config(layout="wide")
st.title("📚 Multi-Modal Researcher AI")

st.sidebar.header("API Configuration")
api_url_input = st.sidebar.text_input(
    "Cloud Run API Invoke URL (e.g., https://<service-url>/research_agent/invoke)",
    value=API_INVOKE_URL
)

st.sidebar.markdown("""
This Streamlit app calls a backend API (deployed on Cloud Run) that uses LangGraph and Gemini
to perform research, synthesize information, and generate a podcast.
""")

st.header("Research Input")
topic = st.text_input("Enter the research topic:", placeholder="e.g., The future of AI in education")
video_url = st.text_input("Optional: Enter a YouTube URL for video analysis:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Configuration section (optional, could be expanded)
# st.subheader("Advanced Configuration (Optional)")
# search_model_override = st.text_input("Override Search Model (e.g., gemini-1.5-pro)", "")
# tts_model_override = st.text_input("Override TTS Model (e.g., gemini-1.5-flash-tts-preview)", "")


if st.button("🚀 Start Research & Generate Podcast"):
    if not api_url_input:
        st.error("Please provide the API Invoke URL in the sidebar.")
    elif not topic:
        st.error("Please enter a research topic.")
    else:
        payload = {
            "input": {
                "topic": topic
            },
            "config": {}, # Initialize empty config
            # "kwargs": {} # Usually not needed for basic invoke
        }
        if video_url:
            payload["input"]["video_url"] = video_url

        # Example of adding runtime configuration if we had more inputs for it
        # configurable_config = {}
        # if search_model_override:
        #     configurable_config["search_model"] = search_model_override
        # if tts_model_override:
        #     configurable_config["tts_model"] = tts_model_override
        # if configurable_config:
        #     payload["config"]["configurable"] = configurable_config

        try:
            with st.spinner("🔬 Performing research, synthesizing report, and generating podcast... This may take a minute or two."):
                response = requests.post(api_url_input, json=payload, timeout=300) # 5 min timeout

            st.subheader("📈 API Response")
            if response.status_code == 200:
                response_data = response.json()

                # The actual output from the graph is usually under an "output" key
                # And intermediate steps might be under "messages"
                output_data = response_data.get("output")

                if output_data:
                    st.balloons()
                    st.success("Research and podcast generation complete!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📝 Research Report")
                        st.markdown(output_data.get("report", "No report generated."))

                    with col2:
                        st.subheader("💬 Podcast Script")
                        st.text_area("Script", value=output_data.get("podcast_script", "No script generated."), height=300)

                        st.subheader("🎙️ Podcast Audio")
                        podcast_audio_url = output_data.get("podcast_url")
                        if podcast_audio_url:
                            st.audio(podcast_audio_url, format="audio/wav")
                        else:
                            st.warning("No podcast audio URL found in the response.")
                else:
                    st.error("Successful API call, but no 'output' data found in the response.")
                    st.json(response_data) # Show full response for debugging

            else:
                st.error(f"API call failed with status code: {response.status_code}")
                try:
                    st.json(response.json()) # Show error response if JSON
                except ValueError:
                    st.text(response.text) # Show raw text if not JSON

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while calling the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Ensure the backend API is deployed and the URL is correctly entered.")
