import zipfile
import xml.etree.ElementTree as ET

with zipfile.ZipFile("assignment.docx", "r") as docx:
    doc_xml = docx.read("word/document.xml").decode("utf-8", errors="ignore")
    root = ET.fromstring(doc_xml)
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }
    for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        p_text = "".join(t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
        for hyper in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink'):
            r_id = hyper.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            print(f"P text: '{p_text}' -> r:id={r_id}")
