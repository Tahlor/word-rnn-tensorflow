import os
import shutil

DIR = r"D:\PyCharm Projects\word-rnn-tensorflow\data\gutenberg"
RESTORE_DIR = r"D:\PyCharm Projects\BACKUP\data\gutenberg"

for root, sub, files in os.walk(DIR):
    for f in files:
        ff = os.path.join(root, f)
        backup = os.path.join(RESTORE_DIR, f)
        #print(backup)
        if os.path.exists(backup):
            print("Restoring {}".format(f))
            shutil.copy(backup, ff)
