
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import textwrap


def create_pdf(image_path, extracted_text, output_path):

    page_w, page_h = A4
    c = canvas.Canvas(output_path, pagesize=A4)

    img = ImageReader(image_path)
    img_w, img_h = img.getSize()
    margin = 15 * mm
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    scale = min(max_w / img_w, max_h / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c.drawImage(img, x, y, width=draw_w, height=draw_h)
    c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, page_h - margin, "Extracted Text")
    c.setFont("Helvetica", 10)

    text_obj = c.beginText(margin, page_h - margin - 25)
    text_obj.setLeading(14)

    if not extracted_text.strip():
        text_obj.textLine("(No text detected)")
    else:
        wrapper = textwrap.TextWrapper(width=95)
        for paragraph in extracted_text.splitlines():
            if not paragraph.strip():
                text_obj.textLine("")
                continue
            for line in wrapper.wrap(paragraph):
                text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()

    return output_path
