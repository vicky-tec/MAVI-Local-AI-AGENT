import os, shutil

DEFAULT_RULES = {
    ".pdf":"Documents", ".docx":"Documents", ".doc":"Documents",
    ".txt":"Documents", ".csv":"Data", ".xlsx":"Sheets",
    ".jpg":"Images", ".jpeg":"Images", ".png":"Images",
    ".mp4":"Videos", ".zip":"Archives"
}

def organize(folder, rules=None):
    rules = rules or DEFAULT_RULES
    moved = 0
    for f in os.listdir(folder):
        src = os.path.join(folder, f)
        if os.path.isfile(src):
            ext = os.path.splitext(f)[1].lower()
            tgt = rules.get(ext)
            if tgt:
                dst = os.path.join(folder, tgt)
                os.makedirs(dst, exist_ok=True)
                try:
                    shutil.move(src, os.path.join(dst, f))
                    moved += 1
                except Exception as e:
                    print("Move error", e)
    return moved
# if __name__ == "__main__":
#     folder = input("Enter folder path to organize: ")
#     if os.path.isdir(folder):
#         count = organize(folder)
#         print(f"Organized {count} files in '{folder}'")
#     else:
#         print("Invalid folder path.")
    