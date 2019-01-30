
import os

# rename files
def rename_items_in_dir(path):
    file_lst = os.listdir(path)

    for idx, file in enumerate(file_lst):
        os.rename(os.path.join(path, file), str(idx) + '.png')

if __name__ == "__main__":
    rename_items_in_dir('validation')

    for valid_png in os.listdir('validation'):
        with open 
    
