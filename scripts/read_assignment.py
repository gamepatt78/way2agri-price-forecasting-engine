import zipfile
import xml.etree.ElementTree as ET

# First read the text export if we fetch it and save with utf-8
import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request('https://docs.google.com/document/d/1axicxD57r_FV4cRXc6o-wPBbEc4eywwK/export?format=txt', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as resp:
    text = resp.read().decode('utf-8')

with open("assignment_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Exported text length:", len(text))
print("--- Text Content ---")
print(text)
