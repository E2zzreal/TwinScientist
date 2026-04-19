# ingestion/audio_processor.py
"""Convert audio recordings into transcripts and extract persona features."""
from ingestion.persona_extractor import (
    extract_style_exemplars,
    extract_verbal_habits,
    extract_reasoning_patterns,
    merge_into_persona,
)


def transcribe_audio(audio_path: str, language: str = "zh") -> str:
    """Transcribe audio file using Whisper. Requires whisper to be installed."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed.\n"
            "Install it with: pip install openai-whisper\n"
            "Or provide transcript_text directly to process_audio()."
        )
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def process_audio(client, model: str, persona_dir: str, speaker: str,
                  audio_path: str | None = None,
                  transcript_text: str | None = None,
                  save_transcript_to: str | None = None) -> dict:
    """Full pipeline: audio (or transcript) → persona features.

    Either audio_path or transcript_text must be provided.
    If audio_path is given, Whisper transcribes it first.
    """
    if transcript_text is None and audio_path is None:
        raise ValueError("Either audio_path or transcript_text must be provided.")

    # Transcribe if needed
    if transcript_text is None:
        transcript_text = transcribe_audio(audio_path)

    # Optionally save transcript
    if save_transcript_to:
        with open(save_transcript_to, "w", encoding="utf-8") as f:
            f.write(transcript_text)

    # Extract features
    exemplars = extract_style_exemplars(client, model, transcript_text, speaker)
    habits = extract_verbal_habits(client, model, transcript_text, speaker)
    patterns = extract_reasoning_patterns(client, model, transcript_text, speaker)

    # Merge into persona files
    merge_into_persona(
        persona_dir,
        new_exemplars=exemplars,
        verbal_habits=habits,
        reasoning_patterns=patterns,
    )

    return {
        "exemplars_added": len(exemplars),
        "verbal_habits_extracted": bool(habits),
        "reasoning_patterns_extracted": bool(patterns),
        "transcript_length": len(transcript_text),
    }