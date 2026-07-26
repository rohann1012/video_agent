import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "whisper-large-v3-turbo"


def transcribe_chunk(chunk_path: str) -> str:
    """
    Transcribe a single audio chunk using Groq Whisper API.
    """

    with open(chunk_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(chunk_path), audio_file.read()),
            model=MODEL_NAME,
            temperature=0,
            response_format="json"
        )

    return transcription.text


def transcribe_all(chunks: list) -> str:
    """
    Transcribe all chunks and combine into a single transcript.
    """

    full_transcript = ""

    print(f"Using {MODEL_NAME} for transcription...\n")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk)

        full_transcript += text + " "

    print("\nTranscription complete.")

    return full_transcript.strip()

