# Helper function to read in tables from the annotations
from html import escape

from bs4 import BeautifulSoup as bs


def format_html(img_html):
    """Formats HTML code from tokenized annotation of img"""
    html_code = img_html["structure"]["tokens"].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ("<td>", ">")]
    for i, cell in zip(to_insert[::-1], img_html["cells"][::-1]):
        if cell["tokens"]:
            cell = [
                escape(token) if len(token) == 1 else token for token in cell["tokens"]
            ]
            cell = "".join(cell)
            html_code.insert(i + 1, cell)
    html_code = "".join(html_code)
    html_code = (
        """<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
                   }
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>"""
        % html_code
    )

    soup = bs(html_code)
    html_code = soup.prettify()
    return html_code
