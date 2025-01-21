import sys
import whisper

model = whisper.load_model("tiny.en")


def transcribe(audio_path):
    result = model.transcribe(audio_path)
    print()
    print("output")
    print("------")
    print(result["text"])
    print()


if __name__ == "__main__":
    print()
    if len(sys.argv) != 2:
        print("missing audio file path!!!")
        print("exiting app.")
        exit(1)
    transcribe(sys.argv[1])
