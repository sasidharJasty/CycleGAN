import os
import shutil

# Root folder containing your images
root_folder = "raw-img"

# Destination folder where all files will be gathered
gathered_folder = "gathered_files"
os.makedirs(gathered_folder, exist_ok=True)

# Extensions to include (add more if needed)
valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}

# Walk through all subdirectories
for dirpath, _, filenames in os.walk(root_folder):
    for file in filenames:
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            src = os.path.join(dirpath, file)
            dest = os.path.join(gathered_folder, file)

            # If a file with the same name exists, add a counter
            base, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(gathered_folder, f"{base}_{counter}{ext}")
                counter += 1

            shutil.copy2(src, dest)
            print(f"Copied: {src} → {dest}")

print("✅ All files have been gathered!")
