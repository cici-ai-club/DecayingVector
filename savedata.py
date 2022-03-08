# Name: train model file
# Command: python savedata.py
# Meaning of Command above: extract train data and test data into default directory
# Author: Chengxi Li
# ======================
import pandas as pd
trainfile = "newevents.csv" # newevents.csv with the cluster labels
from collections import Counter
traindf = pd.read_csv(trainfile) # load the cluster file

def processrow(row, col):
    segList = []
    colist = []
    for c in col:
        if c == "User ID" or c=="Timestamp" or c=='Step' or c=="End Time" or c=="Start Time" or c=='Reasoning Type' or c == 'Reasoning Text' or c=='seg' or c=='Document Name':
            continue # some columns we do not need
        elif "Origin" in c or "Destination" in c:
            if pd.isnull(row[c]):
                continue
            else:
                # get city in the list
                if isinstance(row[c], str):
                    if "{" in row[c]: 
                        city = row[c].split("city': ")[1].split(",")[0].lower().strip("'")
                        if city not in segList:
                            segList.append(city)
                    else:
                        if row[c].lower() not in segList:
                            segList.append(row[c].lower())
        else:
            if pd.isnull(row[c]):
                continue
            else:
                if c =='cls':
                    segList.append("class"+str(int(row[c])))
                else:
                    # get no null column into the segments list
                    segList.append(row[c])
        colist.append(c)
    #print(colist)
    return " ".join(segList) # return all concepts
directory = "./clssave" # the files directory to save
import os

def prepare_train():
    mystack =[]
    train_input = []
    train_target = []
    textdic = {} 
    ob_text = {"text":[],"usr":[],"id":[]}
    for index, row in traindf.iterrows():
        # get pair of curr and next to input and output
        if pd.isnull(row['cls']):
            continue # skip those rows we do not have clusters
        curr = processrow(row, traindf.columns.tolist()) # process every row of data
        usrid = row['User ID']
        ob_text["text"].append(" ".join(curr.split()[:-1]+["eos"]).lower()) # get text
        ob_text['usr'].append(usrid) # collect usr id
        # collect text for each usr into textdic and ob_text
        if usrid not in textdic:
            textdic[usrid]=[curr]
            ob_text['id'].append(str(usrid)+"_"+str(0))
        else:
            if curr!= textdic[usrid][-1]:
                textdic[usrid].append(curr)
            lastid = int(ob_text['id'][-1].split("_")[1])
            ob_text['id'].append(str(usrid)+"_"+str(lastid+1))

    newlm_dic = {"text":[],'usr':[],'label':[]}
    for k,v in textdic.items():
        textdic[k] = " | ".join(v) # get all text of user k into a string
        newlm_dic['text'].append(textdic[k])
        newlm_dic['usr'].append(k)
        newlm_dic['label'].append(textdic[k])

    newlmdf = pd.DataFrame(newlm_dic) 
    # prepare train and test
    for k in  textdic: # k is the test user
        lmtestpre = newlmdf[newlmdf.usr==k]
        lmtrainpre = newlmdf[newlmdf.usr!=k] 
        lmtrain_dic = {"text":[],"label":[],'usr':[],'id':[]}
        for trainind, row in lmtrainpre.iterrows():
            usr = row['usr']
            txt = row['text']
            txtsplit = txt.split(" | ")
            # get input data, label, user for training data
            for index, t in enumerate(txtsplit): 
                if index==0:
                    continue
                else:

                    lmtrain_dic["text"].append(" ".join(txtsplit[index-1].split()[:-1])) # input
                    lmtrain_dic["label"].append(t.split()[-1]) # label
                    lmtrain_dic["usr"].append(usr) # user 
                    lmtrain_dic["id"].append(str(usr)+"_"+str(index))

         # get input data, label, id for test data
        lmtest_dic = {"text":[],"label":[],'usr':[],'id':[]}
        for testind, row in lmtestpre.iterrows():
            usr = row['usr']
            txt = row['text']
            txtsplit = txt.split(" | ")
            for index, t in enumerate(txtsplit):
                if index==0:
                    continue
                else:
                    lmtest_dic["text"].append(" ".join(txtsplit[index-1].split()[:-1]+["eos"]).lower())
                    lmtest_dic["label"].append(t.split()[-1])
                    lmtest_dic["usr"].append(usr)
                    lmtest_dic["id"].append(str(usr)+"_"+str(index))

        # convert the train, test into dataframe and save them
        lmtest = pd.DataFrame(lmtest_dic) 
        lmtrain = pd.DataFrame(lmtrain_dic)
        lmtest.to_csv(os.path.join(directory,"lmtest"+str(k)+".csv"),index=False)
        lmtrain.to_csv(os.path.join(directory,"lmtrain"+str(k)+".csv"),index=False)
    ob_text_df = pd.DataFrame(ob_text)
    ob_text_df.to_csv(os.path.join(directory,"obtext.csv"),index=False)
prepare_train() # prepare the training data
