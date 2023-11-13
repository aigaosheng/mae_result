from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document

i_pdf = [
    'DS3-assessment-Swiss-Dormant-Assets.pdf',
    'DS3-assessment-UK-Dormant-Assets.pdf'
]
o_docs = [
    'Swiss-Dormant-Assets.txt',
    'UK-Dormant-Assets.txt',
]

# convert pdf to txt
for vf, vdoc in zip(i_pdf, o_docs):
    pdf_document = Document(
        document_path=vf,
        language='en'
        )
    pdf2text = PDF2Text(document=pdf_document)
    content = pdf2text.extract()
    with open(vdoc, "wt") as o_fp:
        for vv in content:
            o_fp.write(vv["text"])
