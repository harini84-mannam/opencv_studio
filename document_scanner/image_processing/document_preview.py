from PIL import Image, ImageDraw, ImageFont
import os
import textwrap


def _wrap_line_to_width(draw, line, font, max_width):
    if not line:
        return [""]

    words = line.split(" ")
    wrapped = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textlength(candidate, font=font)

        if width <= max_width or not current:
            current = candidate
        else:
            wrapped.append(current)
            current = word

    
        while draw.textlength(current, font=font) > max_width and len(current) > 1:

            cut = len(current)
            while cut > 1 and draw.textlength(current[:cut], font=font) > max_width:
                cut -= 1
            wrapped.append(current[:cut])
            current = current[cut:]

    if current:
        wrapped.append(current)

    return wrapped or [""]


def create_text_preview(text, output_dir, job_id):
    width = 800
    height = 1100
    margin = 60
    max_text_width = width - (2 * margin)
    line_height = 34

    img = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            22
        )
    except Exception:
        font = ImageFont.load_default()

    x = margin
    y = margin

    for raw_line in text.splitlines():
        for wrapped_line in _wrap_line_to_width(draw, raw_line, font, max_text_width):
            if y > height - margin:
                break
            draw.text(
                (x, y),
                wrapped_line,
                fill="black",
                font=font
            )
            y += line_height
        if y > height - margin:
            break

    filename = f"{job_id}_preview.png"

    path = os.path.join(
        output_dir,
        filename
    )

    img.save(path)

    return filename