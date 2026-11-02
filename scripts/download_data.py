import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

files = {
    "dictionary": "1KW_QZ2vlBz-DM9IIPS05Ql3Ti5LyPD6N",
    "datasource": "1eqXRD_XzuUcQvz5YWGCkRdnC2GWXtDhk"
}

for label, file_id in files.items():
    urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv",
        f"https://docs.google.com/document/d/{file_id}/export?format=docx",
        f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    ]
    success = False
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                content = resp.read()
                ct = resp.headers.get("Content-Type", "")
                cd = resp.headers.get("Content-Disposition", "")
                print(f"[{label}] URL: {url} -> Status: {resp.status}, CT: {ct}, CD: {cd}, Len: {len(content)}")
                if "html" not in ct and len(content) > 100:
                    ext = ".bin"
                    if "officedocument.spreadsheetml" in ct or "xlsx" in url:
                        ext = ".xlsx"
                    elif "csv" in ct or "csv" in url:
                        ext = ".csv"
                    elif "officedocument.wordprocessingml" in ct or "docx" in url:
                        ext = ".docx"
                    elif "text/plain" in ct or "txt" in url:
                        ext = ".txt"
                    elif "pdf" in ct:
                        ext = ".pdf"
                    
                    filename = f"{label}{ext}"
                    with open(filename, "wb") as f:
                        f.write(content)
                    print(f"--> Saved as {filename}!")
                    success = True
                    break
        except Exception as e:
            print(f"[{label}] URL: {url} -> Error: {e}")
    if not success:
        print(f"Failed to cleanly download {label}")
