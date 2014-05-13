'''
Created on Oct 30, 2013

@author: bhikadiy
'''
import json
data = json.load(open('/Jatin/MS/CS545/Project/data/s24-gammasys-gifford-unimpaired.json','r'))
import numpy as np
import matplotlib.pyplot as plt
import random
from PyML import *


def summarize(datalist):
    for i,element in enumerate(datalist):
        keys = element.keys()
        print '\nData set', i
        keys.remove('eeg')
        for key in keys:
            print '  {}: {}'.format(key,element[key])
        eegtrials = element['eeg']
        shape = np.array(eegtrials['trial 1']).shape
        print ('  eeg: {:d} trials, each a matrix with {:d} rows' +
              ' and approximately {:d} columns').format( \
            len(eegtrials), shape[0], shape[1])

def plot(datalist):
    first = datalist[1]
    eeg = np.array(first['eeg']['trial 1'])
    print eeg.shape
    print eeg[8].shape
    np.savetxt("foo.csv", np.array(eeg[8]), delimiter=",")
    plt.figure(1);
    plt.plot(eeg[:,4000:4512].T);
    plt.axis('tight');
    plt.show()
    plt.close()

def plot2(datalist):
    letterp = datalist[4]
    print letterp['protocol']
    eeg = np.array(letterp['eeg']['trial 1']).T
    print eeg.shape
    #plt.ion()
    plt.figure()
    plt.plot(eeg[5000:6000,:9] + 80*np.arange(8,-1,-1))
    plt.plot(np.zeros((1000,9)) + 80*np.arange(8,-1,-1),'--',color='gray')
    plt.yticks([])
    plt.legend(letterp['channels'] + ['Stimulus'])
    plt.axis('tight')
    plt.show()
    plt.close()
    print np.unique(eeg[:,8])
    
def segment(datalist,average):
    for i in datalist:
        print i['protocol']
    positive = []
    negative = []
    all_samples = []
    all_labels = []
    for i in range(6):
        trial = datalist[i+1]
        trial['protocol']
        eeg = np.array(trial['eeg']['trial 1']).T
        print "sample rate : " + str(100.0 / trial['sample rate'])
        starts = np.where(np.diff(np.abs(eeg[:,-1])) > 0)[0]
        stimuli = [chr(int(n)) for n in np.abs(eeg[starts+1,-1])]
        print stimuli
        targetLetter = trial['protocol'][-1]
        print "target letter : " + targetLetter
        targetSegments = (np.array(stimuli) == targetLetter)
        print targetSegments
        for i,(s,stim,targ) in enumerate(zip(starts,stimuli,targetSegments)):
            print s,stim,targ,'; ',
            if (i+1) % 5 == 0:
                print 
        print np.min(np.diff(starts))
        indices = np.array([ np.arange(s,s+210) for s in starts ])
        print indices
        segments = eeg[indices,:8]
        positive_temp = segments[targetSegments,:,:]
        negative_temp = segments[targetSegments==False,:,:]
        print segments.shape
        print positive_temp.shape
        print negative_temp.shape
        positive.append(positive_temp)
        negative.append(negative_temp)
        all_samples.append(positive_temp)
        all_samples.append(negative_temp)
        all_labels.append(np.ones(len(positive_temp)))
        all_labels.append(np.zeros(len(negative_temp)))
    print len(positive)
    print len(negative)
    print len(all_samples)
    print len(all_labels)
    all_samples = np.concatenate(all_samples)
    all_labels = np.concatenate(all_labels)
    positive = np.concatenate(positive)
    print "positives " + str(positive.shape)
    negative = np.concatenate(negative)
    print "negatives" + str(negative.shape)
    print "samples " + str(all_samples.shape)
    print all_labels
    return all_samples,all_labels,positive,negative
        
def classify_with_cv(samples,labels):
    print "inside classify"
    print "samples: " + str(samples.shape)
    np.array(map(str,labels))
    bala = []
    for i in range(8):
        data = VectorDataSet(samples[:,:,i],L=labels)
        print data
        s=SVM()
        r=s.cv(data,5)
        bala.append( r.getBalancedSuccessRate())
    return bala
        
def classify_without_cv(positive,negative):
    train_positive,test_positive = split(positive,5)
    train_negative,test_negative = split(negative,5)
    train = np.concatenate((train_positive,train_negative),axis=0)
    test = np.concatenate((test_positive,test_negative),axis=0)
    train_labels = []
    train_labels.append(np.ones(train_positive.shape[0]))
    train_labels.append(np.zeros(train_negative.shape[0]))
    test_labels = []
    test_labels.append(np.ones(test_positive.shape[0]))
    test_labels.append(np.zeros(test_negative.shape[0]))
    train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)
    np.array(map(str,train_labels))
    np.array(map(str,test_labels))
    predicted_labels = []
    
    for i in range(8):
        train_data = VectorDataSet(train[:,:,i],L=train_labels)
        test_data = VectorDataSet(test[:,:,i],L=test_labels)
        s=SVM()
        s.train(train_data)
        r = s.test(test_data)
        predicted_labels.append(r.getPredictedLabels())
    final_prediction = []
    for i in range(len(predicted_labels[0])):
        decide = []
        for j in range(8):
            decide.append(predicted_labels[j][i])
        one = decide.count(1.0)
        if one>4:
            final_prediction.append(1)
        elif one<4:
            final_prediction.append(0)
        else:
            final_prediction.append(random.randint(0,1))
    print final_prediction
    print test_labels
    TP=0
    TN=0
    FP=0
    FN=0
    P=0
    N=0
    
    for i,j in zip(test_labels,final_prediction):
        if i==1:
            P=P+1
            if j==1:
                TP = TP+1
            if j==0:
                FN=FN+1
        elif i==0:
            N=N+1
            if j==1:
                FP=FP+1
            if j==0:
                TN=TN+1
    print P,N,TP,FN,FP,TN
    BalancedSuccessRate = (float(TP)/float(P) + float(TN)/float(N))/2
    return BalancedSuccessRate
          
def discLDA(X,mu, Sigma, prior):
    Xc=X
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    sigmaInv = np.linalg.pinv(Sigma)
    return np.dot((np.dot(Xc,sigmaInv)),mu) \
    -0.5*np.dot((np.dot(mu.T,sigmaInv)),mu) \
    +np.log(prior)          

def makeStandardizeF(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0,ddof=1)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)
      
def classify_without_cv_lda(positive,negative):
    train_positiv,test_positiv = split(positive,5)
    train_negativ,test_negativ = split(negative,5)
    
    predicted_labels=[]
    for i in range(8):
        train_positive = train_positiv[:,:,i]
        test_positive = test_positiv[:,:,i]
        train_negative = train_negativ[:,:,i]
        test_negative = test_negativ[:,:,i]
        
        
        #train = np.concatenate((train_positive,train_negative),axis=0)
        train = np.vstack((train_positive,train_negative))

        standardize,_ = makeStandardizeF(train)
        trains = standardize(train)
        
        train_positive = trains[0:len(train_positive),:]
        train_negative = trains[len(train_positive):len(trains),:]
        print trains.shape
        print train_positive.shape
        print train_negative.shape
        positive_mu = np.mean(train_positive,axis=0)
        negative_mu = np.mean(train_negative,axis=0)
        
        p_positive = float(len(train_positive))/(float(len(train_positive)+len(train_negative)))
        p_negative = float(len(train_negative))/(float(len(train_positive)+len(train_negative)))
        print p_positive
        print p_negative
        
        sigma_positive = np.cov(train_positive.T)
        sigma_negative = np.cov(train_negative.T)
        
        sigma = sigma_positive*p_positive + sigma_negative*p_negative
        #testing
        test = np.concatenate((test_positive,test_negative),axis=0)
        tests = standardize(test)
        pre1 = discLDA(tests,positive_mu,sigma,p_positive)
        pre2 = discLDA(tests,negative_mu,sigma,p_negative)
        
        #print pre1
        #print pre2
        
        pred = np.argmax(np.vstack((pre2,pre1)), axis=0)
        #print pred
        predicted_labels.append(pred)
#
#     train = np.concatenate((train_positive,train_negative),axis=0)
#     test = np.concatenate((test_positive,test_negative),axis=0)
#     train_labels = []
#     train_labels.append(np.ones(train_positive.shape[0]))
#     train_labels.append(np.zeros(train_negative.shape[0]))
    test_labels = []
#     
    test_labels.append(np.ones(test_positive.shape[0]))
    test_labels.append(np.zeros(test_negative.shape[0]))
#     train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)
#     np.array(map(str,train_labels))
#     np.array(map(str,test_labels))
#     predicted_labels = []
#     
#     for i in range(8):
#         train_data = VectorDataSet(train[:,:,i],L=train_labels)
#         test_data = VectorDataSet(test[:,:,i],L=test_labels)
#         s=SVM()
#         s.train(train_data)
#         r = s.test(test_data)
#         predicted_labels.append(r.getPredictedLabels())
    final_prediction = []
    for i in range(len(predicted_labels[0])):
        decide = []
        for j in range(8):
            decide.append(predicted_labels[j][i])
        one = decide.count(1.0)
        if one>4:
            final_prediction.append(1)
        elif one<4:
            final_prediction.append(0)
        else:
            final_prediction.append(random.randint(0,1))
    print final_prediction
    print test_labels
    TP=0
    TN=0
    FP=0
    FN=0
    P=0
    N=0
     
    for i,j in zip(test_labels,final_prediction):
        if i==1:
            P=P+1
            if j==1:
                TP = TP+1
            if j==0:
                FN=FN+1
        elif i==0:
            N=N+1
            if j==1:
                FP=FP+1
            if j==0:
                TN=TN+1
    print P,N,TP,FN,FP,TN
    BalancedSuccessRate = (float(TP)/float(P) + float(TN)/float(N))/2
    return BalancedSuccessRate          

def classify_channel_lda(positive,negative):
    train_positiv,test_positiv = split(positive,5)
    train_negativ,test_negativ = split(negative,5)
    predicted_labels=[]
    bala = []
    for i in range(8):
        train_positive = train_positiv[:,:,i]
        test_positive = test_positiv[:,:,i]
        train_negative = train_negativ[:,:,i]
        test_negative = test_negativ[:,:,i]
        
        
        #train = np.concatenate((train_positive,train_negative),axis=0)
        train = np.vstack((train_positive,train_negative))

        standardize,_ = makeStandardizeF(train)
        trains = standardize(train)
        
        train_positive = trains[0:len(train_positive),:]
        train_negative = trains[len(train_positive):len(trains),:]
        #print trains.shape
        #print train_positive.shape
        #print train_negative.shape
        positive_mu = np.mean(train_positive,axis=0)
        negative_mu = np.mean(train_negative,axis=0)
        
        p_positive = float(len(train_positive))/(float(len(train_positive)+len(train_negative)))
        p_negative = float(len(train_negative))/(float(len(train_positive)+len(train_negative)))
        #print p_positive
        #print p_negative
        
        sigma_positive = np.cov(train_positive.T)
        sigma_negative = np.cov(train_negative.T)
        
        sigma = sigma_positive*p_positive + sigma_negative*p_negative
        #testing
        test = np.concatenate((test_positive,test_negative),axis=0)
        tests = standardize(test)
        pre1 = discLDA(tests,positive_mu,sigma,p_positive)
        pre2 = discLDA(tests,negative_mu,sigma,p_negative)
        
        #print pre1
        #print pre2
        
        pred = np.argmax(np.vstack((pre2,pre1)), axis=0)
        #print pred
        predicted_labels.append(pred)

        test_labels = []
    #     
        test_labels.append(np.ones(test_positive.shape[0]))
        test_labels.append(np.zeros(test_negative.shape[0]))
        test_labels = np.concatenate(test_labels)

        TP=0
        TN=0
        FP=0
        FN=0
        P=0
        N=0
        #print test_labels
        for i,j in zip(test_labels,pred):
            if i==1:
                P=P+1
                if j==1:
                    TP = TP+1
                if j==0:
                    FN=FN+1
            elif i==0:
                N=N+1
                if j==1:
                    FP=FP+1
                if j==0:
                    TN=TN+1
        print P,N,TP,FN,FP,TN
        BalancedSuccessRate = (float(TP)/float(P) + float(TN)/float(N))/2
        bala.append( BalancedSuccessRate) 
    print "bala " + str(bala)
    return bala
                   
def split(data,parts):
    random.shuffle(data)
    limit=int(len(data)/parts)
    train=data[0:4*limit,:,:]
    test=data[4*limit:len(data),:,:]
    return train,test
    
if __name__ == '__main__':
    print "hello"
    summarize(data)
    #plot(data)
    #plot2(data)
    samples,labels,positives,negatives = segment(data,1)
    
#     balc=[]
#     for i in range(20):
#         balc.append(classify_with_cv(samples,labels))
#     balcc=np.array(balc)
#     print balcc.shape
#     final = []
#     for i in range(balcc.shape[1]):
#         temp_rate = balcc[:,i]
#         print sum(temp_rate)/20.0
#         final.append(sum(temp_rate)/20.0)
#     print final

    balc=[]
    for i in range(6):
        balc.append(classify_channel_lda(positives,negatives))
    balcc=np.array(balc)
    print balcc.shape
    print balcc
    final = []
    for i in range(balcc.shape[1]):
        temp_rate = balcc[:,i]
        print sum(temp_rate)/10.0
        final.append(sum(temp_rate)/10.0)
    print final

#     
#     balc=[]
#     for i in range(1):
#         balc.append(classify_without_cv_lda(positives,negatives))
#     balcc=np.array(balc)
#     print balcc.shape
#     print balcc
#     print sum(balcc)

#     balc=[]
#     for i in range(1):
#         balc.append(classify_without_cv(positives,negatives))
#     balcc=np.array(balc)
#     print balcc.shape
#     print balcc
#     print sum(balcc)    
#     
#classify_without_cv(positives,negatives)
    