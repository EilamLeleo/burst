#!/usr/lib/python-exec/python2.7/python
import os
import sys

from neuron import h
from neuron import gui

#%%

import numpy as np
import time
import math
import cPickle as pickle

#%%

sk = False

if sk==True:
    from sklearn import decomposition
    from sklearn import cluster
    from sklearn import linear_model
    from sklearn import ensemble
    from sklearn import cross_validation

#%%

h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

cvode = h.CVode()
cvode.active(1)

morphologyFilename = "morphologies/cell1.asc"
#morphologyFilename = "morphologies/cell2.asc"
#morphologyFilename = "morphologies/cell3.asc"

#biophysicalModelFilename = "L5PCbiophys1.hoc"
#biophysicalModelFilename = "L5PCbiophys2.hoc"
#biophysicalModelFilename = "L5PCbiophys3.hoc"
#biophysicalModelFilename = "L5PCbiophys4.hoc"
#biophysicalModelFilename = "L5PCbiophys5.hoc"
biophysicalModelFilename = "L5PCbiophys5b.hoc"

#biophysicalModelTemplateFilename = "L5PCtemplate.hoc"
biophysicalModelTemplateFilename = "L5PCtemplate_2.hoc"


#%%

h.load_file(biophysicalModelFilename)
h.load_file(biophysicalModelTemplateFilename)
L5PC = h.L5PCtemplate(morphologyFilename)

#%% set dendritic VDCC g=0
#secs = h.allsec

#L5PC.soma.gSKv3_1bar_SKv3_1 = 0.338*1.5

VDCC_g = 1

if VDCC_g==0:
    for sec in h.allsec():
        if hasattr(sec, 'gCa_HVAbar_Ca_HVA'):
            sec.gCa_HVAbar_Ca_HVA = 0
    

#%% helper functions 

def Add_NMDA_SingleSynapticEventToSegment(segment, activationTime, synapseWeight):

#    synapse = h.ProbAMPANMDA_EMS(segLoc,sec=section)
    synapse = h.ProbAMPANMDA2(segment)
    synapse.gmax = .0004
    synapse.Use = 1.0
    synapse.Dep = 0
    synapse.Fac = 0

    netStimulation = h.NetStim()                   
    netStimulation.number = 1
    netStimulation.start = activationTime
    
    netConnection = h.NetCon(netStimulation,synapse)
    netConnection.delay = 0
    netConnection.weight[0] = synapseWeight

    return netStimulation,netConnection,synapse

#%% create length-weighted random section list

def randSecWeight(obj,medSeg,part,num):
    allLen = []
    for i in range(len(obj)):
        allLen.append(obj[i].L)
    randSecList = [0 for i in range(num)]
    h.distance(sec=obj[medSeg]) # define distance measure from medSeg
    # draw from cumulative length a seg for syn
    x = np.sum(allLen[:medSeg])+(np.random.rand(num)-0.5)*np.sum(allLen)/part
    j=0
    farbug=0
    while j<num:
        # redraw boundary crossers
        if x[j]<0 or x[j]>np.sum(allLen):
            x[j] = np.sum(allLen[:medSeg])+(np.random.rand()-0.5)*np.sum(allLen)/part
            continue
        # find sec
        for i in range(len(obj)):
            if x[j]<np.sum(allLen[:i+1]):
                randSecList[j]=i
                break
        # check that sec is sufficiently close to medseg
        if h.distance(obj[randSecList[j]](1))>sum(allLen)/part and farbug<5:#obj[medSeg].L+obj[randSecList[j]].L:#
            x[j] = np.sum(allLen[:medSeg])+(np.random.rand()-0.5)*np.sum(allLen)/part
            farbug+=1
            continue
        j+=1
        farbug=0
    return randSecList

#%% add some random NMDA synapses and plot a somatic trace just to see all things are alive and kicking

def runSim(cell,ApiBasInt,treeT,numBas,numApi,partApi,medSeg,numExp):
    
    simulationTime    = 400
    silentTimeAtStart = 100
    delayTime = 200
    silentTimeAtEnd   = 100
    
    
    origNumSamplesPerMS = 20
    totalSimDuration = simulationTime + silentTimeAtStart + silentTimeAtEnd
        
    listOfSomaTraces = []
    spikes = []
    numSpikes = 0
    numSpikesPerExp = [0]*numExp
    freq = [0]*numExp
    
    for experiment in range(numExp):
    
        startTime = time.time()
                
        listOfRandBasalSectionInds  = randSecWeight(cell.dend,44,1,int(numBas))#np.random.randint(0,len(cell.dend),int(numBas)) 
        listOfRandApicalSectionInds = randSecWeight(cell.apic,np.random.randint(37,78),partApi,int(numApi)) 
#        listOfRandObliqueSectionI2nds = np.random.randint(0,len(cell.apic)/partApi,0)#int(40-numApi)) #obliques
        listOfBasalSections  = [cell.dend[x] for x in listOfRandBasalSectionInds]
        listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
#        listOfObliqueSections = [cell.apic[x] for x in listOfRandObliqueSectionInds]
        #listOfApicalSections = []
        
#        listOfSections = listOfApicalSections + listOfBasalSections
        
        listOfRandBasalLocationsInSection = np.random.rand(len(listOfRandBasalSectionInds))
        listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
#        listOfRandObliqueLocationsInSection = np.random.rand(len(listOfRandObliqueSectionInds))
#        listOfSegLocs = list(listOfRandApicalLocationsInSection) + list(listOfRandBasalLocationsInSection)
        
        listOfEvents = []
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 1))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 1))
        
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + delayTime + treeT*np.random.normal(0,1) #gauss(0.5,0.2)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 1))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + delayTime + ApiBasInt + treeT*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 1))
        
        #add obliques
#        for k, section in enumerate(listOfObliqueSections):
#            eventTime = silentTimeAtStart + delayTime + treeT*np.random.normal(1,0.2) #simulationTime/2*np.random.rand(1)[0]
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandObliqueLocationsInSection[k]), eventTime, 2))
        
        
        ##%% run the simulation
        recTime = h.Vector()
        recTime.record(h._ref_t)
        recVoltage = h.Vector()
        recVoltage.record(cell.soma[0](0.5)._ref_v)
        
        cvode.cache_efficient(1)
        h.finitialize(-76)
        stopTime = totalSimDuration 
        neuron.run(stopTime)
        
        # plot the trace
        origRecordingTime = np.array(recTime.to_python()) # ugly workaround to recTime.as_numpy()
        origSomaVoltage = np.array(recVoltage.to_python()) # ugly workaround to recVoltage.as_numpy()
        
        recordingTime = np.arange(0,totalSimDuration,1.0/origNumSamplesPerMS)
        somaVoltage   = np.interp(recordingTime, origRecordingTime, origSomaVoltage)    
        listOfSomaTraces.append(somaVoltage)
        
        
        origSpikes = []
        tempSpikes = 0
        
        k = (silentTimeAtStart+delayTime-50)*origNumSamplesPerMS #int(np.min([0,ApiBasInt]))
        while k < (totalSimDuration-silentTimeAtEnd)*origNumSamplesPerMS:
            if somaVoltage[k]>-10:
                tempTime = float(k)/origNumSamplesPerMS
                if tempSpikes>0 and tempTime-origSpikes[-1]>20:
                    break
                origSpikes.append(tempTime)
                # numSpikesPerExp[experiment] = tempSpikes + 1
                numSpikes = numSpikes + 1
                tempSpikes += 1 # numSpikesPerExp[experiment]
                k = k+origNumSamplesPerMS*3
            else:
                k = k+5 # was 1 before
        
    #    spikes = []
        spikes.append(origSpikes)
        if tempSpikes>1: 
            freq[experiment] = tempSpikes/(origSpikes[-1]-origSpikes[-tempSpikes])
        
        plt.figure('trc api30+bas50 dt0 sd?')
        plt.plot(recordingTime, somaVoltage)
        plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
        plt.axis(xmin=0, xmax=stopTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
        
        #listOfEvents = []
        if (experiment+1)%10==0 or (time.time()-startTime)/60>5 or numExp<5: 
            print "Dt %s treeTime %s exp. # %s took %.3f minutes" % (ApiBasInt,treeT,experiment+1, (time.time()-startTime)/60)
        
    print "Mean no. of spikes: %s" % (float(numSpikes)/numExp)
    return float(numSpikes)/numExp,np.mean(freq)#, listOfSomaTraces, recordingTime

#%% run simulation on some parameter pair, plot the space
L5PC = h.L5PCtemplate(morphologyFilename)
name = 'dttreet_pt10api35meds60'
saveDir = '/ems/elsc-labs/segev-i/eilam.goldenberg/Documents/coincidence/wgh1/'+name+'/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

try:
    randomSeed = int(sys.argv[1])
    print 'random seed selected by user - %d' %(randomSeed)
except:
    randomSeed = np.random.randint(100000)
    print 'randomly chose seed - %d' %(randomSeed)

np.random.seed(randomSeed)

#ind = 1
#a = np.linspace(-50,-25,num=6),np.linspace(-20,20,num=21),np.linspace(25,100,num=16)
ApicalBasalInterval = np.linspace(-40,60,num=21)#51) #[x for xs in a for x in xs]
numBasal = 50 #np.linspace(0,200,num=81)
numApical = 30 #np.linspace(0,100,num=41)#50,num=21)#
#numOblique = 30-numApical
#totalSyn = [i for i in np.linspace(10,100,num=10)]+[i for i in np.linspace(150,500,num=8)]
#[20,50,100,200,400,600,800]#[80,120,150,180]#np.linspace(0,200,num=5)#41)
partApical = 10 #np.logspace(0,7,num=29,base=2)
medSegment = 60 #np.linspace(0,80,num=9)
treeTime = np.linspace(1,20,num=20)#0.1*np.logspace(3,10,num=15,base=2)
numExperiments = 20

spks = [[0 for i in range(len(ApicalBasalInterval))] for j in range(len(treeTime))] 
frqs = [[0 for i in range(len(ApicalBasalInterval))] for j in range(len(treeTime))]
#trc = [[[] for i in range(len(ApicalBasalInterval))] for j in range(len(treeTime))]

i = 0
j = 0

start = time.time()

for ApiBasInd in ApicalBasalInterval:
    print "Running for dt: %s [ms]" % (ApiBasInd)
#for numB in numBasal:#totalS in totalSyn:#
#    print "Running for %s basal synapses" % (int(numB))
    for treeT in treeTime:#partApi in partApical:
#    for numA in numApical:#np.linspace(0,totalS,num=41):#
        print "Running for treeTime: %s [ms]" % (treeT) #apical synapses" % (int(numA))
#        numA = int(totalS*0.4)
        spks[j][i],frqs[j][i] = runSim(L5PC,ApiBasInd,treeT,numBasal,numApical,partApical,medSegment,numExperiments)
        j = j+1
    j = 0
    i = i+1
    
pickle.dump(spks,open(saveDir+name+'_spks'+str(randomSeed)+".npy","wb"),protocol=2)
pickle.dump(frqs,open(saveDir+name+'_frqs'+str(randomSeed)+".npy","wb"),protocol=2)

print "Saved as ", saveDir+name+'_spks'+str(randomSeed)+".npy"
print "Total running time was: ", (time.time()-start)/3600, "hours"

#saveDir = '/ems/elsc-labs/segev-i/eilam.goldenberg/Documents/concidence/'
#pickle.dump(spks1,open(saveDir+'dt_treet_30tot_hires_spks',"wb"),protocol=2)
#pickle.dump(frqs1,open(saveDir+'dt_treet_30tot_hires_frqs',"wb"),protocol=2)
