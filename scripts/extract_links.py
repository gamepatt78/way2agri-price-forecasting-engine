import zipfile
import xml.etree.ElementTree as ET

with zipfile.ZipFile("assignment.docx", "r") as docx:
    for name in docx.namelist():
        if "rels" in name:
            content = docx.read(name).decode("utf-8", errors="ignore")
            print(f"=== {name} ===")
            # find all Target="..."
            root = ET.fromstring(content)
            for elem in root.iter():
                target = elem.attrib.get("Target")
                if target:
                    print("Rel:", elem.attrib.get("Id"), "->", target)

    print("\n=== word/document.xml hyperlinks ===")
    doc_xml = docx.read("word/document.xml").decode("utf-8", errors="ignore")
    root = ET.fromstring(doc_xml)
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    }
    for hyper in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink'):
        r_id = hyper.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        text = "".join(t.text for t in hyper.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
        print(f"Link text: '{text}', r:id: {r_id}")
