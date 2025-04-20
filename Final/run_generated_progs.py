import os

pathToTests = "tests"
files = os.listdir(pathToTests)

print(files)

for file in files:
    print(file)
    with open(pathToTests+"/"+file) as script:
        print("Running "+file)
        exec(script.read())