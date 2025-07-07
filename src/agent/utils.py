import os
import wave
import datetime # For signed URL expiration
from google.genai import Client, types
from google.cloud import storage # For GCS operations
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

# Initialize client
gemini_api_key_value = os.getenv("GEMINI_API_KEY")

# !!! WARNING: DEBUGGING CODE - RE-ADDED - REMOVE AFTER TESTING !!!
if gemini_api_key_value:
    print(f"DEBUG (utils.py - Initialization): GEMINI_API_KEY loaded. Length: {len(gemini_api_key_value)}, First 5 chars: {gemini_api_key_value[:5]}", flush=True)
else:
    print("DEBUG (utils.py - Initialization): GEMINI_API_KEY IS NOT SET or is empty! Client initialization might fail or use default credentials if available.", flush=True)
# !!! END OF DEBUGGING CODE !!!

genai_client = Client(api_key=gemini_api_key_value)


def display_gemini_response(response):
    """Extract text from Gemini response and display as markdown with references"""
    console = Console()
    
    # Extract main content
    text = response.candidates[0].content.parts[0].text
    md = Markdown(text)
    console.print(md)
    
    # Get candidate for grounding metadata
    candidate = response.candidates[0]
    
    # Build sources text block
    sources_text = ""
    
    # Display grounding metadata if available
    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
        console.print("\n" + "="*50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("="*50)
        
        # Display and collect source URLs
        if candidate.grounding_metadata.grounding_chunks:
            console.print(f"\n[bold]Sources ({len(candidate.grounding_metadata.grounding_chunks)}):[/bold]")
            sources_list = []
            for i, chunk in enumerate(candidate.grounding_metadata.grounding_chunks, 1):
                if hasattr(chunk, 'web') and chunk.web:
                    title = getattr(chunk.web, 'title', 'No title') or "No title"
                    uri = getattr(chunk.web, 'uri', 'No URI') or "No URI"
                    console.print(f"{i}. {title}")
                    console.print(f"   [dim]{uri}[/dim]")
                    sources_list.append(f"{i}. {title}\n   {uri}")
            
            sources_text = "\n".join(sources_list)
        
        # Display grounding supports (which text is backed by which sources)
        if candidate.grounding_metadata.grounding_supports:
            console.print(f"\n[bold]Text segments with source backing:[/bold]")
            for support in candidate.grounding_metadata.grounding_supports[:5]:  # Show first 5
                if hasattr(support, 'segment') and support.segment:
                    snippet = support.segment.text[:100] + "..." if len(support.segment.text) > 100 else support.segment.text
                    source_nums = [str(i+1) for i in support.grounding_chunk_indices]
                    console.print(f"• \"{snippet}\" [dim](sources: {', '.join(source_nums)})[/dim]")
    
    return text, sources_text


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM data to a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def create_podcast_discussion(topic, search_text, video_text, search_sources_text, video_url, filename="research_podcast.wav", configuration=None):
    """Create a 2-speaker podcast discussion explaining the research topic"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()
    
    # Step 1: Generate podcast script
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_pd_script = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_pd_script:
        print(f"DEBUG (create_podcast_discussion - Script Gen - Runtime): About to call Gemini. Key starts with: {gemini_api_key_runtime_pd_script[:5]}", flush=True)
    else:
        print("DEBUG (create_podcast_discussion - Script Gen - Runtime): GEMINI_API_KEY not found in env at runtime!", flush=True)
    # --- END DEBUG ---
    script_prompt = f"""
    Create a natural, engaging podcast conversation between Dr. Sarah (research expert) and Mike (curious interviewer) about "{topic}".
    
    Use this research content:
    
    SEARCH FINDINGS:
    {search_text}
    
    VIDEO INSIGHTS:
    {video_text}
    
    Format as a dialogue with:
    - Mike introducing the topic and asking questions
    - Dr. Sarah explaining key concepts and insights
    - Natural back-and-forth discussion (5-7 exchanges)
    - Mike asking follow-up questions
    - Dr. Sarah synthesizing the main takeaways
    - Keep it conversational and accessible (3-4 minutes when spoken)
    
    Format exactly like this:
    Mike: [opening question]
    Dr. Sarah: [expert response]
    Mike: [follow-up]
    Dr. Sarah: [explanation]
    [continue...]
    """
    
    script_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=script_prompt,
        config={"temperature": configuration.podcast_script_temperature}
    )
    
    podcast_script = script_response.candidates[0].content.parts[0].text
    
    # Step 2: Generate TTS audio
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_pd_tts = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_pd_tts:
        print(f"DEBUG (create_podcast_discussion - TTS Gen - Runtime): About to call Gemini for TTS. Key starts with: {gemini_api_key_runtime_pd_tts[:5]}", flush=True)
    else:
        print("DEBUG (create_podcast_discussion - TTS Gen - Runtime): GEMINI_API_KEY not found in env at runtime for TTS!", flush=True)
    # --- END DEBUG ---
    tts_prompt = f"TTS the following conversation between Mike and Dr. Sarah:\n{podcast_script}"
    
    response = genai_client.models.generate_content(
        model=configuration.tts_model,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker='Mike',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=configuration.mike_voice,
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker='Dr. Sarah',
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=configuration.sarah_voice,
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )
    
    # Step 3: Save audio file
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(filename, audio_data, configuration.tts_channels, configuration.tts_rate, configuration.tts_sample_width)
    print(f"Podcast saved locally as: {filename}")

    # Step 4: Upload to GCS and generate signed URL
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        print("GCS_BUCKET_NAME environment variable not set. Skipping GCS upload.")
        # Fallback: In a real scenario, you might want to handle this more gracefully
        # or make GCS upload mandatory. For now, we'll return None for the URL.
        # Alternatively, if running locally without GCS, one might want to serve the local file.
        # However, for Cloud Run, GCS is the way for persistent, accessible files.
        return podcast_script, None # Or raise an error

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)

        # Sanitize filename for GCS path if necessary, though current filename is likely fine
        # The filename already includes topic, making it somewhat unique.
        # Adding a timestamp or UUID could make it more robustly unique if needed.
        blob_name = f"podcasts/{filename}"
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(filename)
        print(f"Uploaded {filename} to gs://{gcs_bucket_name}/{blob_name}")

        # Generate a signed URL for the blob, valid for 1 hour
        expiration_time = datetime.timedelta(hours=1)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        print(f"Generated signed URL: {signed_url}")

        # Clean up local file after upload (optional, good for stateless environments)
        try:
            os.remove(filename)
            print(f"Removed local file: {filename}")
        except OSError as e:
            print(f"Error removing local file {filename}: {e}")

        return podcast_script, signed_url
    except Exception as e:
        print(f"Error during GCS upload or signed URL generation: {e}")
        # Fallback or error handling
        return podcast_script, None # Or re-raise the error after logging


def create_research_report(topic, search_text, video_text, search_sources_text, video_url, configuration=None):
    """Create a comprehensive research report by synthesizing search and video content"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()

    # Step 1: Create synthesis using Gemini
    # --- DEBUG: Runtime API Key Check ---
    gemini_api_key_runtime_report = os.getenv("GEMINI_API_KEY")
    if gemini_api_key_runtime_report:
        print(f"DEBUG (create_research_report - Synthesis - Runtime): About to call Gemini. Key starts with: {gemini_api_key_runtime_report[:5]}", flush=True)
    else:
        print("DEBUG (create_research_report - Synthesis - Runtime): GEMINI_API_KEY not found in env at runtime!", flush=True)
    # --- END DEBUG ---
    synthesis_prompt = f"""
    You are tasked with producing a high-quality academic literature review on the topic of "{topic}".
    Your review should be based *solely* on the information provided from the 'SEARCH RESULTS' and 'VIDEO CONTENT' sections below.
    Do not invent external information or sources.

    The literature review should be structured, analytical, and comprehensive. Please adhere to the following guidelines:

    1.  **Title:** Start with a clear, descriptive title for the literature review. (e.g., "A Literature Review on {topic}")

    2.  **Introduction (1-2 paragraphs):**
        *   Briefly introduce the topic of "{topic}".
        *   State the purpose and scope of this literature review (i.e., to synthesize and analyze the provided information).
        *   Outline the main themes or areas that will be covered in the review.

    3.  **Thematic Analysis / Key Concepts (Multiple Paragraphs, organize by themes):**
        *   Identify the major themes, concepts, findings, or arguments presented in the 'SEARCH RESULTS' and 'VIDEO CONTENT'.
        *   For each theme/concept:
            *   Provide a detailed explanation and synthesis of the information.
            *   Critically analyze the information: discuss its significance, compare and contrast different points if they exist in the provided materials.
            *   Clearly indicate which insights come from the search results versus the video content, if discernible and relevant.
        *   Ensure a logical flow between themes. Use transition sentences to connect ideas.

    4.  **Discussion (1-2 paragraphs):**
        *   Provide an overall discussion of the topic based on the synthesized information.
        *   Highlight any significant patterns, trends, or consistencies you observed in the provided materials.
        *   If the provided materials suggest any limitations, gaps, or areas for further inquiry (based *only* on what's given), mention them. Do not speculate beyond the provided text.

    5.  **Conclusion (1 paragraph):**
        *   Summarize the main findings of your literature review.
        *   Reiterate the key insights derived from the provided 'SEARCH RESULTS' and 'VIDEO CONTENT'.
        *   Offer a final concluding thought on the topic of "{topic}" based on the review.

    6.  **Tone and Style:**
        *   Maintain a formal, objective, and academic tone throughout the review.
        *   Use precise language. Avoid jargon where possible, or explain it if necessary (based on provided content).
        *   Ensure clarity, coherence, and logical organization.

    7.  **Length and Depth:**
        *   Aim for a comprehensive and in-depth review. The length should be dictated by the depth of analysis the provided materials allow, rather than a specific word or paragraph count, but should be substantially more detailed than a brief summary. Strive for at least 6-8 well-developed paragraphs, or more if the content supports it.

    **INPUT MATERIALS:**

    SEARCH RESULTS:
    {search_text}

    VIDEO CONTENT:
    {video_text}

    ---
    Begin the literature review now, starting with the title.
    """
    
    synthesis_response = genai_client.models.generate_content(
        model=configuration.synthesis_model,
        contents=synthesis_prompt,
        config={
            "temperature": configuration.synthesis_temperature,
        }
    )
    
    synthesis_text = synthesis_response.candidates[0].content.parts[0].text
    
    # Step 2: Create markdown report string
    report_content = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}

## Additional Sources
{search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""

    # Step 3: Upload report to GCS
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        print("GCS_BUCKET_NAME environment variable not set. Skipping GCS upload for report.")
        # In a real scenario, decide how to handle this. For now, returning content and None URL.
        return report_content, synthesis_text # Or return None, synthesis_text if URL is mandatory

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)

        # Create a unique filename for the report
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        report_filename = f"research_report_{safe_topic.replace(' ', '_')}.md"
        blob_name = f"reports/{report_filename}"
        blob = bucket.blob(blob_name)

        # Upload the report content
        blob.upload_from_string(report_content, content_type='text/markdown')
        print(f"Uploaded report to gs://{gcs_bucket_name}/{blob_name}")

        # Generate a signed URL for the blob, valid for 1 hour
        expiration_time = datetime.timedelta(hours=1)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        print(f"Generated signed URL for report: {signed_url}")

        return signed_url, synthesis_text
    except Exception as e:
        print(f"Error during GCS upload or signed URL generation for report: {e}")
        # Fallback or error handling
        return report_content, synthesis_text # Or None, synthesis_text