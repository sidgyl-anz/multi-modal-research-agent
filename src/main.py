from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional, Dict, Any
import os

# Assuming the graph and state definitions are accessible
# Adjust imports based on your project structure
from agent.graph import create_compiled_graph
from agent.state import ResearchStateInput
from agent.configuration import Configuration

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Multi-Modal Research Agent API",
    description="API to trigger research, analysis, and podcast generation.",
    version="0.1.0"
)

# Initialize the compiled LangGraph app
# This assumes create_compiled_graph() doesn't require immediate config or can use defaults
# If config is needed at startup, adjust accordingly
try:
    research_app = create_compiled_graph()
except Exception as e:
    print(f"Error initializing compiled graph: {e}")
    research_app = None

# --- Pydantic Models for Request and Response ---
class ResearchRequest(BaseModel):
    topic: str
    video_url: Optional[HttpUrl] = None
    create_podcast: bool = True
    recipient_email: EmailStr

class ResearchResponse(BaseModel):
    message: str
    report_url: Optional[str] = None
    podcast_url: Optional[str] = None
    synthesis_text: Optional[str] = None # Exposing this as it's a key output

# --- Placeholder for Email Sending Function ---
# This will be implemented in a later step
async def send_research_email_background(
    recipient_email: EmailStr,
    topic: str,
    report_url: Optional[str],
    podcast_url: Optional[str]
):
    print(f"Background task: Simulating sending email to {recipient_email} for topic '{topic}'")
    print(f"Report URL: {report_url}")
    print(f"Podcast URL: {podcast_url}")
    # Actual email sending logic will go here
    # For now, this is just a print statement.
    # You'll need to integrate an email library like smtplib, sendgrid, etc.
    # and handle SMTP configurations.
    # --- Actual Email Sending Logic ---
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_str = os.getenv("SMTP_PORT", "587")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_sender_email = os.getenv("SMTP_SENDER_EMAIL")
    smtp_use_tls_str = os.getenv("SMTP_USE_TLS", "True").lower()
    smtp_use_ssl_str = os.getenv("SMTP_USE_SSL", "False").lower()

    if not all([smtp_host, smtp_port_str, smtp_username, smtp_password, smtp_sender_email]):
        print("SMTP configuration is incomplete. Skipping email.")
        # In a real app, you might want to log this more formally or raise an alert
        return

    try:
        smtp_port = int(smtp_port_str)
        smtp_use_tls = smtp_use_tls_str in ("true", "1")
        smtp_use_ssl = smtp_use_ssl_str in ("true", "1")

        # Create the email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your Research Results for: {topic}"
        msg["From"] = smtp_sender_email
        msg["To"] = recipient_email

        text_content = f"""
        Hello,

        Here are the results for your research topic: "{topic}"

        Report URL: {report_url if report_url else "Not generated"}
        Podcast URL: {podcast_url if podcast_url else "Not generated / Not requested"}

        Thank you for using the Research Agent.
        """

        html_content = f"""
        <html>
        <head></head>
        <body>
            <h2>Research Results for: {topic}</h2>
            <p>Hello,</p>
            <p>Here are the results for your research topic: <strong>{topic}</strong></p>
            <p><strong>Report URL:</strong> {f'<a href="{report_url}">{report_url}</a>' if report_url else "Not generated"}</p>
            <p><strong>Podcast URL:</strong> {f'<a href="{podcast_url}">{podcast_url}</a>' if podcast_url else "Not generated / Not requested"}</p>
            <br>
            <p>Thank you for using the Research Agent.</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        if smtp_use_ssl:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)

        if smtp_use_tls and not smtp_use_ssl: # STARTTLS is not used if SSL is already active
            server.starttls()

        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {recipient_email} for topic '{topic}'")

    except Exception as e:
        print(f"Failed to send email to {recipient_email} for topic '{topic}'. Error: {e}")
        # In a real app, consider more robust error handling/logging or retry mechanisms

@app.post("/research/", response_model=ResearchResponse)
async def run_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    if not research_app:
        raise HTTPException(status_code=503, detail="Research service is unavailable. Graph not initialized.")

    print(f"Received research request: {request.model_dump()}")

    # Prepare input for the LangGraph agent
    agent_input: ResearchStateInput = {
        "topic": request.topic,
        "video_url": str(request.video_url) if request.video_url else None,
        "create_podcast": request.create_podcast,
        # recipient_email is handled by the API layer after graph execution
    }

    # RunnableConfig can be used to pass dynamic configurations if needed
    # For now, using default/environment-based config loaded by Configuration class
    config = {"configurable": Configuration().model_dump()}

    try:
        # Invoke the LangGraph agent
        # The output type of research_app.invoke should match ResearchStateOutput structure
        # but we only need specific fields for the response.
        result = research_app.invoke(agent_input, config=config)

        # Assuming result is a dictionary matching ResearchStateOutput structure
        report_url = result.get("report")
        podcast_url = result.get("podcast_url")
        synthesis_text = result.get("synthesis_text") # Assuming this is available from the state

        # Add email sending to background tasks
        # The email function itself will be implemented properly later
        background_tasks.add_task(
            send_research_email_background,
            request.recipient_email,
            request.topic,
            report_url,
            podcast_url
        )

        return ResearchResponse(
            message="Research process completed. Email will be sent with results.",
            report_url=report_url,
            podcast_url=podcast_url,
            synthesis_text=synthesis_text
        )

    except Exception as e:
        print(f"Error during research agent invocation: {e}")
        # Consider more specific error handling based on exception types
        raise HTTPException(status_code=500, detail=f"An error occurred during the research process: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080)) # Default to 8080, configurable via PORT env var
    uvicorn.run(app, host="0.0.0.0", port=port)
