import os
import subprocess
import sys

def check_for_decomp():
    with open("temp.txt") as output:
        lines = output.readlines()
        foundBefore = 0
        foundBeforeDef = 0
        foundAfter = 0
        foundAfterDef = 0
        scalarFunction = 0
        beforeLines = []
        afterLines = []
        if(len(lines) == 0):
            print("Error, file empty")
        else:
            #Scan through lines
            for line in lines:
                if(scalarFunction == 0):
                    if "Scalar-returning" in lines:
                        scalarFunction = 1
                    elif "Before" in line:
                        foundBefore = 1
                    elif "After" in line:
                        foundAfter = 1
                        foundBeforeDef = 0
                    elif "def forward" in line and foundBefore == 1 and foundAfter == 0:
                        foundBeforeDef = 1
                    elif "def forward" in line and foundAfter == 1:
                        foundAfterDef = 1
                    elif foundBeforeDef:
                        beforeLines.append(line.strip())
                    elif foundAfterDef:
                        afterLines.append(line.strip())
                else: #Sclar function
                    if "is NOT decomposed" in lines:
                        print("Decomp failed")
                        return 0
                    elif "is decomposed" in lines:
                        print("Decomp worked")
                        return 1
                #print("FB: "+str(foundBefore)+"FA: "+str(foundAfter)+"FBD: "+str(foundBeforeDef)+"FAD: "+str(foundAfterDef))
            #Check if function are the same or different
            if(scalarFunction == 0):
                differingLines = 0
                for i in range(min(len(beforeLines),len(afterLines))):
                    if (beforeLines[i] != afterLines[i]):
                        differingLines += 1
                if(differingLines>0):
                    print("Decomp worked: expanded from "+str(len(beforeLines))+" to "+str(len(afterLines))+" lines with at least "+str(differingLines)+" changed.")
                    return 1
                else:
                    print("Decomp failed. All lines the same.")
                    return 0

pathToTests = "tests"
files = os.listdir(pathToTests)

python_executable = sys.executable  # This points to the correct Python (conda env)

errorCount = 0
ranCount = 0
decompWorkedCount = 0

for i in range(5):  # Or len(files) for all tests
    file_path = os.path.join(pathToTests, files[i])
    print(f"Running {files[i]}")

    try:
        result = subprocess.run(
            [python_executable, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        with open("temp.txt", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"{files[i]} caused an error.")
            errorCount += 1 
        else:
            print(f"{files[i]} ran without exception.")
            ranCount += 1
            decompWorkedCount += check_for_decomp()
    except Exception as e:
        print(f"Failed to run {files[i]} due to: {e}")

print(f"{ranCount}/{ranCount+errorCount} ran without eror. {decompWorkedCount}/{ranCount} decomps worked.")




# import os
# import sys

# pathToTests = "tests"
# files = os.listdir(pathToTests)

# #print(files)

# # Save standard output (will be overwritten by the test file)
# original_std_out = sys.stdout

# # for file in files:
# #     print(file)
# for i in range(4): #len(files)
#     with open(pathToTests+"/"+files[i]) as script:
#         print("Running "+files[i])
#         exception = False
#         try:
#             sys.stdout = open("temp.txt", "w")
#             exec(script.read())
#         except:
#             exception = True
#         finally:
#             sys.stdout = original_std_out
#         if(exception):
#             print(files[i]+" caused an error.")
#         else:
#             print(files[i]+" ran without exception.")
#             pass
            # 
