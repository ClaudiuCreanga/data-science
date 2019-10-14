import os


directory = os.fsencode("/Users/claudiucreanga/Downloads/england")

count = 1
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     os.rename(os.path.join("/Users/claudiucreanga/Downloads/england/", filename), os.path.join("/Users/claudiucreanga/Downloads/england/", str(count) + '.jpg'))
     count += 1
