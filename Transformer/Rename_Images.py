import os

def rename_files(folder_path):
    i = 82
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            old_filepath = os.path.join(folder_path, filename)
            new_filename = f"{i}.jpg"
            new_filepath = os.path.join(folder_path, new_filename)
            
            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {old_filepath} to {new_filepath}")
            except FileExistsError:
                print(f"File {new_filepath} already exists.")
            
            i += 1

if __name__ == "__main__":
    folder_path = r"D:/HAU Lab/Local Work 2/Test Data By Class Renamed/use_screw_driver"

    rename_files(folder_path)

