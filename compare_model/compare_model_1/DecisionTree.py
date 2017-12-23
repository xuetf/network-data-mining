from math import log
file = open("/Users/iEric/Downloads/txt/training_result.txt")
featurefile = open("/Users/iEric/Downloads/txt/feature0.txt")
fline = featurefile.readlines()
featurefile.close()
line = file.readlines()
file.close()
class Node(object):
    def __init__(self, word, Child0, Child1, flag ):
        self.word=word
        self.Child0=Child0
        self.Child1=Child1
        self.flag=flag
    def setChild0(self, Child0):
        self.Child0=Child0
    def setChild1(self, Child1):
        self.Child1=Child1
    def setFlag(self, flag):
        self.flag=flag
    def OutPut(self):
        
        print(self.word+" "+str(self.flag))
        if self.Child1!=None:
            self.Child1.OutPut()
        if self.Child0!=None:
            self.Child0.OutPut()
    
    def writeTreeFromPreorder(self, fileName):
        file = open(fileName,'a')
        file.write(self.word+","+str(self.flag)+"|")
        file.close()
        if self.Child1!=None:
            self.Child1.writeTreeFromPreorder(fileName)
        if self.Child0!=None:
            self.Child0.writeTreeFromPreorder(fileName)
    def writeTreeFromInorder(self, fileName):
        if self.Child1!=None:
            self.Child1.writeTreeFromInorder(fileName)
        file = open(fileName,'a')
        file.write(self.word+","+str(self.flag)+"|")
        file.close()
        if self.Child0!=None:
            self.Child0.writeTreeFromInorder(fileName)

def makeListFromFile(fileNameFisrt,fileNameIn):
    prefile=open(fileNameFisrt)
    preorderstr=prefile.readline()
    prefile.close()
    infile=open(fileNameIn)
    inorderstr=infile.readline()
    infile.close()
    preList=preorderstr.split("|")
    inList=inorderstr.split("|")
    preList.pop()
    inList.pop()
    return(preList,inList)
def buildTreeFromList(preList,i,inList,low,high):
    if i >= len(preList):
        return None
    root=Node(preList[i][:len(preList[i])-2],None,None,int(preList[i][len(preList[i])-1]))
    j=0
    for j in range(low,high+1,1):
        if(preList[i]==inList[j]):
            break;
    leftTree=buildTreeFromList(preList,i+1,inList,low,j-1)
    rightTree=buildTreeFromList(preList,(i+j-low+len(preList)),inList,j+1,high)
    root.setChild1(leftTree)
    root.setChild0(rightTree)
    return root

def findDivideFeature(fline,line):
    word=""
    temp=10000
    pos=-1
    L0=[]
    L1=[]
    for j in range(len(fline)):
        count11 = 0
        count10 = 0
        count01 = 0
        count00 = 0
        h = 10000
        List1=[]
        List0=[]
        s=fline[j].split("'")[1]
        for i in range(len(line)):
            if(line[i][1:].find(s)!=-1):
                if(line[i][0]=='1'):
                    List1.append(line[i])
                    count11=count11+1
                else:
                    List1.append(line[i])
                    count10=count10+1
            else:
                if(line[i][0]=='1'):
                    List0.append(line[i])
                    count01=count01+1
                else:
                    List0.append(line[i])
                    count00=count00+1
        s1=count11+count10
        s0=count01+count00
        if(s1*s0>0):
            h=0
            p11=float(count11)/float(s1)
            p10=float(count10)/float(s1)
            p01=float(count01)/float(s0)
            p00=float(count00)/float(s0)
            if p11>0:
                h11=p11*log(p11)*(-1)
            else:
                h11=0
            if p10>0:
                h10=p10*log(p10)*(-1)
            else:
                h10=0
            if p01>0:
                h01=p01*log(p01)*(-1)
            else:
                h01=0
            if p00>0:
                h00=p00*log(p00)*(-1)
            else:
                h00=0
            h1=h11+h10
            h0=h01+h00
            h=float(s1)/float(s1+s0)*h1+float(s0)/float(s1+s0)*h0
        if(temp>h):
            pos=j
            word=s
            L1=List1
            L0=List0
            temp=h

    return (word,pos,L1,L0)
def deletefeature(word,pos,fline):
    List=[]
    for i in range(len(fline)):
        if i!=pos:
            List.append(fline[i])
    return List
def makeDecisionTreeNode(fline,line):
    A=findDivideFeature(fline,line)
    flag=3
    if len(line)>0:
        flag=int(line[0][0])
    for i in range(len(line)):
        if line[0][0]!=line[i][0]:
            flag=2
            break
    node =None
    if A[0]!="" :
        node=Node(A[0],None,None,flag)
    else:
        node=Node("",None,None,compareSum(line))
    line1=A[2]
    line0=A[3]
    fline0=deletefeature(A[0],A[1],fline)
    return (node, fline0, line1, line0,fline0)
def compareSum(line):
    item=int(line[0][0])
    count=0
    for i in range(len(line)):
        if int(line[i][0])==item:
            count=count+1
    if count*2>len(line):
        return item
    else:
        return 1-item


def makeDecisionTree(fline,line):
    if len(line)==0 :
        return None
    if len(fline)==0 :
        return Node("",None,None,compareSum(line))
    A = makeDecisionTreeNode(fline,line)
    node = A[0]
    if node==None :
        return node
    if node.flag!=2 :
        return node
    node.setChild1(makeDecisionTree(A[1],A[2]))
    node.setChild0(makeDecisionTree(A[1],A[3]))
    return node
def divide(testline,node):
    flagstr=""
    for i in range(len(testline)):
        flagstr=flagstr+str(divideline(testline[i],node))
    return flagstr
def divideline(teststr,node):
    if (node.flag!=2):
        return node.flag
    if (teststr[1:].find(node.word)!=-1):
        if(node.Child1!=None):
            return divideline(teststr,node.Child1)
    if (teststr[1:].find(node.word)==-1):
        if(node.Child0!=None):
            return divideline(teststr,node.Child0)
def calculateAccuracy(testline,teststr):
    count=0
    for i in range(len(testline)):
        if testline[i][:1]==teststr[i]:
            count=count+1
    return float(count)/float(len(testline))
def calculatePrecision(testline,teststr):
    count=0
    sum=0
    for i in range(len(testline)):
        if (testline[i][:1]==teststr[i]) & (teststr[i]=="1"):
            count=count+1
        if teststr[i]=="1":
            sum=sum+1
    return float(count)/float(sum)
def calculateRecall(testline,teststr):
    count=0
    sum=0
    for i in range(len(testline)):
        if (testline[i][:1]==teststr[i]) & (teststr[i]=="1"):
            count=count+1
        if testline[i][:1]=="1":
            sum=sum+1
    return float(count)/float(sum)
def calculateF1(precision,recall):
    return 2*precision*recall/(precision+recall)


def loadModel(FirstTree,InTree):
    Tree=makeListFromFile(FirstTree,InTree)
    return Tree
def predict(wordList,Tree):
    strList=divide(wordList,Tree)
    return strList



'''
A=makeListFromFile("/Users/iEric/Downloads/txt/TreeF.txt","/Users/iEric/Downloads/txt/TreeI.txt")
preList=A[0]
inList=A[1]
node = buildTreeFromList(preList,0,inList,0,len(preList)-1)
node.OutPut()
'''



tfile=open("/Users/iEric/Downloads/txt/testing_classify_result.txt")
testline0 = tfile.readlines()
num=0
nonestr=testline0[1]
testline=[]
testList=[]
for i in range(len(testline0)):
    if testline0[i]!=nonestr:
        testline.append(testline0[i])
        testList.append(testline0[i][0])

tfile.close()

Tree=makeDecisionTree(fline,line)
#Tree.OutPut()
Tree.writeTreeFromPreorder("/Users/iEric/Downloads/txt/TreeF.txt")
Tree.writeTreeFromInorder("/Users/iEric/Downloads/txt/TreeI.txt")
strList=divide(testline,Tree)

#writefile = open("/Users/iEric/Downloads/txt/test.txt",'w')
#teststr=writefile.readline()
#writefile.write(strList)
#writefile.close()






















