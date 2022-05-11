import os

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                if full_filename[-3:] == '.py':
                    print(full_filename)
    except PermissionError:
        pass
    
search("C:/python")