import os
import shutil
import glob

def move_images(source_folder, destination_folder):
    # Creating the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Searching for image files in subfolders
    image_files = glob.glob(os.path.join(source_folder, '**/*.png'), recursive=True) + \
                  glob.glob(os.path.join(source_folder, '**/*.jpg'), recursive=True) + \
                  glob.glob(os.path.join(source_folder, '**/*.jpeg'), recursive=True) + \
                  glob.glob(os.path.join(source_folder, '**/*.gif'), recursive=True) + \
                  glob.glob(os.path.join(source_folder, '**/*.bmp'), recursive=True)

    # Moving the image files to the destination folder
    for image_file in image_files:
        filename = os.path.basename(image_file)
        destination_path = os.path.join(destination_folder, filename)

        # Checking if the destination file already exists
        if not os.path.exists(destination_path):
            shutil.move(image_file, destination_folder)
        else:
            print(f"Skipping '{filename}' - File already exists in the destination folder.")


source_folder = 'drop drill'
destination_folder = 'Drop_Drill'

move_images(source_folder, destination_folder)
