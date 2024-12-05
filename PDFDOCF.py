import pandas as pd
from xlsx2html import xlsx2html
import pdfkit
from io import StringIO

# Convert Excel to HTML
html_stream = StringIO()
xlsx2html('D:\\COLLEGE PROJECT\\startbootstrap-resume-gh-pages\\tESTING.xlsx', html_stream)
html_content = html_stream.getvalue()

# Enhance the HTML content with styles for better formatting
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Excel to PDF</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Save the enhanced HTML content to a temporary file
with open('temp.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

# Specify the path to wkhtmltopdf executable
config = pdfkit.configuration(wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')

try:
    # Convert HTML to PDF
    pdfkit.from_file('temp.html', 'output.pdf', configuration=config)
    print("PDF generated successfully.")
except OSError as e:
    print(f"Error: {e}")
