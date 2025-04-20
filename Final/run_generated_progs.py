import os
import sys

pathToTests = "tests"
files = os.listdir(pathToTests)

print(files)

# Save standard output (will be overwritten by the test file)
std_out = sys.stdout

# for file in files:
#     print(file)
for i in range(len(files)):
    with open(pathToTests+"/"+files[i]) as script:
        print("Running "+files[i])
        exec(script.read())
        sys.stdout = std_out
        functionName = files[i].split("test_")[1].split(".py")[0]
        print("Opening output_"+functionName+".txt")
        with open("output_"+functionName+".txt") as output:
            lines = output.readlines()
            foundBefore = 0
            foundBeforeDef = 0
            foundAfter = 0
            foundAfterDef = 0
            beforeLines = []
            afterLines = []
            if(len(lines) == 0):
                print("Error, file empty")
            else:
                #Scan through lines
                for line in lines:
                    if "Before" in line:
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
                    #print("FB: "+str(foundBefore)+"FA: "+str(foundAfter)+"FBD: "+str(foundBeforeDef)+"FAD: "+str(foundAfterDef))
                #Check if function are the same or different
                differingLines = 0
                for i in range(min(len(beforeLines),len(afterLines))):
                    if (beforeLines[i] != afterLines[i]):
                        differingLines += 1
                if(differingLines>0):
                    print("Decomp worked: expanded from "+str(len(beforeLines))+" to "+str(len(afterLines))+" lines with at least "+str(differingLines)+" changed.")
                else:
                    print("Decomp failed. All lines the same.")
