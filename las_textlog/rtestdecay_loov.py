# Name: test model file
# Command: python rtestdecay_loov.py 0 0.5 
# Meaning of Command above: test model on user 0 with decaying rate 0.5
# Author: Chengxi Li
# ======================
import pandas as pd
import sys
import numpy as np
import torch
from mynet import network
from copy import deepcopy
testusr = int(sys.argv[1])
initdecay = float(sys.argv[2])
import random
random.seed(10)
print("decaying:",initdecay)
print("usr:",testusr)
split = "all"

def mytokenizer(row):
        return row['text'].lower().split()
def specialtokenizer(row):
    row['text'] = str(row['text'])
    return row['text'].lower().split()[0]

def decaytransform(text,word2id,st2int,lastone, decay=initdecay):
    textsplit = text.lower().split()
    tmp = lastone*decay
    for x in textsplit:
        tmp[word2id[x]] = 1
    eachvec = tmp
    return eachvec

# build vector using decaying rate
def builddata(df,word2id, st2int,batch_size=3,mem=None):
    trainset =[] 
    stack = []
    lasttext ='' 
    for index, row in df.iterrows():
        if row['usr'] not in stack:
            inputsvec = decaytransform(row['text'],word2id,st2int,np.zeros(len(word2id)))
            stack.append(row['usr'])
            curr = row['text']
        else:
            inputsvec = decaytransform(row['text'],word2id,st2int,trainset[-1]['inputs'])
            curr = lasttext+" "+row['text'] # track current input text
    
        lasttext = curr
        labelid = st2int[row['label']]
        myfinal = df[df['usr']==row['usr']].label.tolist()[-1] # get final label
        trainset.append({"inputs":inputsvec,"label":labelid,"text":curr,"textlabel":row['label'],"futurelabel":" ".  join(df[df['usr']==row['usr']].loc[index:]['label'].tolist()),'endlabel':myfinal,"usr":row['usr']})
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False,  num_workers=2)
    return trainloader


def testit(testloader,PATH,model):
    model.load_state_dict(torch.load(PATH)) # load the trained model
    model.eval()
    res = {"usr":[],"inputs":[],"label":[],"futurelabel":[],"enddistance":[],"enddiscount":[],'randcorrect':[],'randcorrect_future':[],'randdistance':[],'randdiscount':[],"correct":[],"correct_future":[],"distance":[],"discount":[],"pred":[]}
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs = data["inputs"]
            inputs = inputs.float()
            label = data["label"]
            outputs= model(inputs)
            outputs = torch.sigmoid(outputs) # change output to probability
            outputs_left = torch.nonzero(outputs>=0.5) # using confidence 0.5 to get all satisfying result out
            prob = []
            for o in outputs_left: # get the probabily of filtered output with confidence  0.5
                prob.append(outputs[o[0]][o[1]].item())
            
            sort_output = sorted(zip(outputs_left,prob),key=lambda x:x[1],reverse=True) # sort the output using probabilty
            outputs_left = list(map(lambda x:x[0],sort_output))
            prob = list(map(lambda x:x[1],sort_output))
            predicted = outputs_left
           
            batchsize = len(label)
            predstr = {}
            predicted = {}
            probeach = {}
            for k in range(len(label)):
                predstr[k] = []
                predicted[k] = []
                probeach[k] = []
            for j, b in enumerate(outputs_left):
                predstr[b[0].item()].append(int2st[b[1].item()]) # get predicted  class
                predicted[b[0].item()].append(b[1].item())  # get predicted id
                probeach[b[0].item()].append(prob[j]) # get predicted probability 
            randstr = random.sample(list(st2int.keys()),len(label)) # random sample some predicting string
            randid = [st2int[q] for q in randstr] 
            futurelabel = data["futurelabel"]
            res['usr'] += data["usr"].tolist() # collect users 
            endlabel = data["endlabel"] # collect end labels
            res['inputs'].extend(data['text']) # collect inputs
            res['label'].extend(data['textlabel'])  # collect labels
            res["randcorrect"] += (torch.tensor(randid) == label.cpu()).tolist() # random prediction is correct
            res["randcorrect_future"] += [p in futurelabel[ip].split() for ip, p in enumerate(randstr)] # random prediction is correct in the future
            res["randdistance"] += [futurelabel[ip].split().index(p) if p in futurelabel[ip].split() else -10**12 for ip, p in        enumerate(randstr)] # random prediction distance with label
            res["randdiscount"] += [0.8**(futurelabel[ip].split().index(p)) if p in futurelabel[ip].split() else 0 for ip, p in     enumerate(randstr)] # random discount future score
            res["enddistance"] += [futurelabel[ip].split().index(p) if p in futurelabel[ip].split() else -10**12 for ip, p in  enumerate(endlabel)] # the distance of end prediction
            res["enddiscount"] += [0.8**(futurelabel[ip].split().index(p)) if p in futurelabel[ip].split() else 0 for ip, p in     enumerate(endlabel)]  # end prediction discount future score
            res['futurelabel'].extend(futurelabel) # extract future labels

            # get correct, correct future, distance and discount 
            for k, v in predstr.items():
                if not v:
                    res["correct"].append(-1)
                    res["correct_future"].append(-1)
                    res["pred"].append(None)
                    res["distance"].append(-1)
                    res["discount"].append(-1)
                    continue
                res["pred"].append(list(zip(v,probeach[k])))
                flag = False
                for pv in predicted[k]:
                    if pv == label[k]:
                        flag = True
                        break
                res["correct"].append(int(flag))
                futureflag = False
                tmpdistance = []
                tmpdiscount = []
                for vv in v:
                    if vv in futurelabel[k].split():
                        futureflag = True
                        fv = vv
                        tmpdistance.append(futurelabel[k].split().index(fv))
                        tmpdiscount.append(0.8**(futurelabel[k].split().index(fv)))
                    else:
                        tmpdistance.append(10**12)
                        tmpdiscount.append(0)
                res["distance"].append(min(tmpdistance))
                res["discount"].append(0.8**res["distance"][-1])
                res["correct_future"].append(int(futureflag))

    print("acc:",sum((np.array(res["correct"])!=-1)*np.array(res["correct"]))/sum(np.array(res["correct"])!=-1),"len:",sum(np.array(res["correct"])!=-1))
    print("acc_f:",sum((np.array(res["correct_future"])!=-1)*np.array(res["correct_future"]))/sum(np.array(res["correct_future"])!=-1),"len:",sum(np.array(res["correct_future"])!=-1))
    
    print("discount:",sum((np.array(res["discount"])!=-1)*np.array(res["discount"]))/sum(np.array(res["discount"])!=-1))
    print()

    print("randacc:",sum(res["randcorrect"])/len(res["randcorrect"]),"len:",len(res["randcorrect"]))
    print("randacc_f:",sum(res["randcorrect_future"])/len(res["randcorrect_future"]),"len:",len(res["randcorrect_future"])) 
    print("enddiscount:",sum(res["enddiscount"])/len(res["enddiscount"])) 
    print("randdiscount:",sum(res["randdiscount"])/len(res["randdiscount"])) 


    resdf = pd.DataFrame(res)
    # save the result to local
    resdf.to_csv("saveres_loov/savepredic_res_confidence0.5"+'all'+"_"+str(initdecay)+"_"+str(testusr)+".csv",index=False)

# load the mapping we built in the training file
mappings = np.load('mappings.npy', allow_pickle=True).item()
st2int = mappings["st2int"]
int2st = mappings["int2st"]
word2id = mappings["word2i2"]
id2word = mappings["id2word"]
vocab = set(word2id.keys())
def testoriginal():
    oldtestdf = pd.read_csv("clssave/lmtest"+str(testusr)+".csv") # load test data
    testdf = oldtestdf
    testloader = builddata(testdf,word2id, st2int, batch_size=2) # build data
    model = network(len(vocab),200,len(st2int)) # get model
    print("load "+ "model_clssave/bestmodel"+str(testusr)+"_"+str(initdecay)+".pt")
    testit(testloader,"model_clssave/bestmodel"+str(testusr)+"_"+str(initdecay)+".pt",model) # test out
testoriginal()
