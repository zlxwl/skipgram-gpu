import os

filepath = 'D:\9900\查件'

def read_files(filepath):
    filelist = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            filelist.append(os.path.join(root, file))
    return filelist

filelist = read_files(filepath)
print(filelist)