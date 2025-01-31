import argparse
import heapq
import os
import re

import chromadb
import nltk
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

load_dotenv()
set_llm_cache(SQLiteCache(database_path="summarizer.db"))

model = ChatGroq(model="llama-3.2-3b-preview")


def get_video_id(video_url):
    if not isinstance(video_url, str):
        return False, "Invalid input: URL must be a string."

    video_id_pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_pattern, video_url)

    if not match:
        return False, "Invalid URL: Unable to extract video ID."

    video_id = match.group(1)
    return True, video_id


def get_youtube_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        transcript = transcript_list.find_transcript(["en"]).fetch()
    except Exception:
        transcript = None
        for entry in transcript_list:
            try:
                transcript = entry.translate("en").fetch()
                break
            except Exception:
                pass
        if transcript is None:
            return False, "transcript not found"
    formatter = TextFormatter()
    formatted_transcript = formatter.format_transcript(transcript)
    sanitized_transcript = re.sub(
        r"\[music\]", "", formatted_transcript, flags=re.IGNORECASE
    )
    return True, sanitized_transcript


def summarize_transcript(transcript):
    messages = [
        SystemMessage("Summarize the following text:"),
        HumanMessage(transcript),
    ]
    response = model.invoke(messages)
    return response.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Get YouTube transcript from a video URL."
    )
    parser.add_argument("video_url", type=str, help="The URL of the YouTube video")
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Show only the summary of the transcript",
    )
    args = parser.parse_args()
    video_url = args.video_url
    show_summary = args.summary
    status, video_id = get_video_id(video_url)
    if status is False:
        print(f"error: {video_id}")
        return
    status, transcript = get_youtube_transcript(video_id)
    if status is False:
        print(f"error: {transcript}")
    if show_summary:
        summary = summarize_transcript(transcript)
        print(summary)
    else:
        print(transcript)


if __name__ == "__main__":
    main()
