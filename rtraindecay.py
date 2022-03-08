# Name: train model file
# Command: python rtraindecay.py 0 0.5 
# Meaning of Command above: train model on without user 0 with decaying rate 0.5
# Author: Chengxi Li
# ======================
import pandas as pd
import sys
import numpy as np
import torch
from mynet import network
from copy import deepcopy

initdecay = float(sys.argv[2]) # the decaying rate
print("decaying:",initdecay)
def mytokenizer(row):
    return row['text'].lower().split()
# using decaying rate to build a vector of current input (idea From Dr.Ware)
def decaytransform(text,word2id,st2int,lastone, decay=initdecay):
    textsplit = text.lower().split()
    tmp = lastone*decay
    for x in textsplit:
        tmp[word2id[x]] = 1
    eachvec = tmp 
    return eachvec


def builddata(df,word2id, st2int,batch_size=3): # build input text data into vector and convert label to id
    trainset =[]
    stack = []
    for index, row in df.iterrows():
        if row['usr'] not in stack: # the first one direct use decay algorithm
            inputsvec = decaytransform(row['text'],word2id,st2int,np.zeros(len(word2id)))
            stack.append(row['usr'])
        else:
            inputsvec = decaytransform(row['text'],word2id,st2int,trainset[-1]['inputs']) 


        labelid = st2int[row['label']] # get label id
        trainset.append({"inputs":inputsvec,"label":labelid}) # get current input vector and label id
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    return trainloader



def trainit(trainloader,net,epochs=100): # training
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6) # initiate optimizer
    minloss =10000000000000000
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("epoch:",epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["inputs"]
            inputs = inputs.float()
            label = data["label"]
            label = label
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, loss= net(inputs,labels=label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0
        if running_loss <minloss:
            minloss = running_loss
            print("minloss:",minloss) # keep updating the min loss
            #torch.save(the_model.state_dict(), PATH)
            bestmodel = deepcopy(net.state_dict()) # get the best model with minimum loss
    torch.save(bestmodel, "model_clssave/bestmodel"+str(testusr)+"_"+str(initdecay)+".pt")
    print('Finished Training')

testusr = sys.argv[1] # get the test user
obs_df = pd.read_csv("clssave/obtext.csv") # get all obsevation data
words = obs_df.apply(mytokenizer,axis =1)  # get input text words
vocab = [] # build vocabulary
sortedwords = sorted(list(words.tolist())) # sort words
for x in sortedwords:
    for xx in x:
        if  xx not in vocab:
            vocab.append(xx)
word2id ={w:index for index, w in enumerate(vocab)}
id2word ={index:w for index, w in enumerate(vocab)}
mappings = {"word2i2":word2id,"id2word":id2word} # get mapping betwwen word and index

st2int = {}
int2st = {}

oldtraindf = pd.read_csv("clssave/lmtrain"+str(testusr)+".csv") # get train data
testdf = pd.read_csv("clssave/lmtest"+str(testusr)+".csv") #  get test data
alldf = pd.concat([testdf,oldtraindf],ignore_index=True) # get all data
traindf = oldtraindf
# get label vocab
labelset = set(alldf['label'].tolist())
labelset = list(labelset)
labelset = sorted(labelset,key = lambda x:int(x.split("class")[1]))
for i, l in enumerate(labelset):
    st2int[l] = i
    int2st[i] = l
mappings["st2int"] = st2int
mappings["int2st"] = int2st
np.save('mappings.npy', mappings) # save input text vocab and label vocab into mappings
trainloader = builddata(traindf,word2id, st2int, batch_size=2) # build data
model = network(len(vocab),200,len(st2int)) # build model
trainit(trainloader,model,epochs=3) # train it using data and model
