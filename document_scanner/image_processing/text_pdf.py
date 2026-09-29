from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_text_only_pdf(text, output_path, title="Document"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)

    c.setFont("Helvetica", 11)
    y = height - 80

    for line in text.splitlines():
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 50

    c.save()