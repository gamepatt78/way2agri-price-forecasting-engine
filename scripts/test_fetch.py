import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

urls = [
    'https://docs.google.com/document/d/1axicxD57r_FV4cRXc6o-wPBbEc4eywwK/export?format=txt',
    'https://docs.google.com/document/d/1axicxD57r_FV4cRXc6o-wPBbEc4eywwK/export?format=docx',
    'https://drive.google.com/uc?export=download&id=1axicxD57r_FV4cRXc6o-wPBbEc4eywwK',
    'https://drive.usercontent.google.com/download?id=1axicxD57r_FV4cRXc6o-wPBbEc4eywwK&export=download',
    'https://docs.google.com/document/d/1axicxD57r_FV4cRXc6o-wPBbEc4eywwK/edit?usp=sharing'
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            content = resp.read()
            ct = resp.headers.get('Content-Type')
            print(f"URL: {url[:50]}... -> Status: {resp.status}, Content-Type: {ct}, Length: {len(content)}")
            if "text" in str(ct) or len(content) < 2000:
                print("Text Preview:", content[:500].decode('utf-8', errors='ignore'))
            if "officedocument" in str(ct) or url.endswith("docx"):
                with open("assignment.docx", "wb") as f:
                    f.write(content)
                print("Saved assignment.docx!")
    except Exception as e:
        print(f"URL: {url[:50]}... -> Error: {e}")
