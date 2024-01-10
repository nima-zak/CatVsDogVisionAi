import os
from PIL import Image

def check_images(s_dir, ext_list):
    """Checks images in a directory for corruption and incorrect extensions.

    Args:
        s_dir (str): Path to the directory containing the images.
        ext_list (list): List of allowed image extensions.

    Returns:
        tuple: (list, list)
            - A list of paths to corrupted images.
            - A list of paths to files with incorrect extensions.
    """

    bad_images = []
    bad_ext = []

    for folder in os.listdir(s_dir):
        folder_path = os.path.join(s_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    extension = file.split('.')[-1].lower()
                    if extension in ext_list:
                        try:
                            img = Image.open(file_path)
                            img.verify()  # Verify image integrity
                        except (IOError, SyntaxError) as e:
                            print(f"Corrupted image: {file_path}")
                            bad_images.append(file_path)
                            # Uncomment to delete corrupted images:
                            # os.remove(file_path)
                    else:
                        print(f"Incorrect file extension: {file_path}")
                        bad_ext.append(file_path)

    return bad_images, bad_ext

# Define dataset path and extensions
source_dir = './data'  # Replace with your dataset path
extensions = ['jpg', 'jpeg', 'png', 'gif']  # Add more extensions if needed

# Check images and print results
bad_images, bad_ext_files = check_images(source_dir, extensions)

print(f"Corrupted images: {bad_images}")
print(f"Files with incorrect extensions: {bad_ext_files}")
