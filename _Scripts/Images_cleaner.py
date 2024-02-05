import os

image_dir = 'G:\\_PYTHON_learn\\CONROS\\Images\\'
#18442

count = 0
for path, subdirs, files in os.walk(image_dir):
    for name in files:
        year = name.split('_')[1]
        if len(year) > 4 or '?' in str(year):
            print(year)
            os.remove(path + '\\' + name)
            count += 1
print(count, 'bad images were deleted.')
