import gradio as gr
from PIL import Image

from caption import generate_caption


def caption_image(image):
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occured: {str(e)}"


def main():
    iface = gr.Interface(
        fn=caption_image,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="image aptioning with blip",
        description="upload an image to generate a caption.",
    )
    iface.launch()


if __name__ == "__main__":
    main()
