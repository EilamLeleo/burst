 # -*- coding: utf-8 -*-
"""
Spyder Editor

Script for generating voltage traces
"""

#%% imports etc

import os
import sys
os.chdir('C:/Users/Leleo/Documents/Active Cell Real Morphology/')

from neuron import h
from neuron import gui

import numpy as np
import time
import math
import cPickle as pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")
h.celsius = 34

cvode = h.CVode()
cvode.active(0)

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

h.load_file(biophysicalModelFilename)
h.load_file(biophysicalModelTemplateFilename)
L5PC = h.L5PCtemplate(morphologyFilename)

def Add_NMDA_SingleSynapticEventToSegment(segment, activationTime, synapseWeight, exc_inh):

    if exc_inh==0:     # inhibitory
        synapse = h.ProbGABAAB_EMS(segment) #GABAA/B
        synapse.tau_r_GABAA = 0.2
        synapse.tau_d_GABAA = 8
        synapse.tau_r_GABAB = 3.5
        synapse.tau_d_GABAB = 260.9
#        synapse.gmax = .001
        synapse.e_GABAA = -80
        synapse.e_GABAB = -97
        synapse.GABAB_ratio = 0.0

    else:           # excitatory
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

def randSecDist(obj,dist,part,num):
    randSecList = [0 for i in range(num)]
    h.distance(sec=L5PC.soma[0]) # define distance measure from medSeg
    # draw from cumulative length a seg for syn
    maxDist = 1300#max(h.distance(obj[i](1)) for i in range(len(obj)))
    j=0
#    farbug=0
    while j<num:
        # find sec
        i=0
        while i<len(obj)*10:
            x=int(np.random.rand(1)*len(obj))
            if h.distance(obj[x](1))>dist-maxDist/part/2 and h.distance(obj[x](0))<dist+maxDist/part/2:
                randSecList[j]=x
                break
            i+=1
        j+=1
    return randSecList

#%% Predefining variables given to function runSim: cell,ApiBasInt,treeT,numBas,numApi,partApi,medSeg,numExp
cell = L5PC
ApiBasInt = [0]#np.linspace(-30,30,num=61)#0 #15 # 20 # 40
treeT = 0 #9.05 #4.53 #2.26 #
numApi = 10#; ind=0#15 #80 #100 #5 # INH
numA = [0,10,20]#,30]#,40,50,80,100] #Exc Api
numBas = 100 #90 #35 #
partApi = 20 #
dist = np.linspace(100,1300,num=7)
medSegment = [0,4,36,60,61,62,63,26] #26 #20 #14 #4 #0 #63 #
medSeg= 61 #medSegment[-2]
numExp = len(ApiBasInt)

cell.soma[0].gSKv3_1bar_SKv3_1 = 0.338*1.5#2#
#%% TTX
for i in range(len(cell.axon)):
    cell.axon[i].gNap_Et2bar_Nap_Et2 = 0
#    cell.axon[i].gNaTa_tbar_NaTa_t = 0

#%% Copied from Burst.py

L5PC = h.L5PCtemplate(morphologyFilename)
cell = L5PC
cell.soma[0].gSKv3_1bar_SKv3_1 = 0.338*1.5#2#

simulationTime    = 400
silentTimeAtStart = 100
delayTime = 200
#    ApicalBasalInterval = 0 #60
#    treeTime = 20
silentTimeAtEnd   = 100

origNumSamplesPerMS = 20
totalSimDuration = simulationTime + silentTimeAtStart + silentTimeAtEnd
    
#for experiment in range(numExp):

startTime = time.time()

listOfBackgroundBasalSecInds = np.random.randint(0,len(cell.dend),int(numBas/2*(1+np.random.normal(0,0.1))))#int(numBas))#
listOfBackgroundApicalSecInds = np.random.randint(0,len(cell.apic),int(numA[-1]/2*(1+np.random.normal(0,0.1))))#int(numApi))#
listOfRandBasalSectionInds  = randSecWeight(cell.dend,44,1,int(numBas))#np.random.randint(0,len(cell.dend),int(numBas))
listOfRandApicalSectionInds = randSecWeight(cell.apic,62,partApi,numA[-1])#medSeg + np.random.randint(-distance,distance,int(numApi))
listOfRandInhSectionInds = [int(medSeg)]*int(numApi)#randSecDist(cell.apic,medSeg,20,numApi)# + np.random.randint(0,len(cell.apic)/partApi,numApi)
#for clust in range(int(partApi)):
#    np.append(listOfRandApicalSectionInds, np.random.randint(0,len(cell.apic),1) 
#                                    + np.random.randint(0,len(cell.apic)/partApi,int(numApi/partApi)), axis=0)
#            medS = medS + 128/partApi
listOfBgBasalSec  = [cell.dend[x] for x in listOfBackgroundBasalSecInds]
listOfBgApicalSec = [cell.apic[x] for x in listOfBackgroundApicalSecInds]
listOfBasalSections  = [cell.dend[x] for x in listOfRandBasalSectionInds]
listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
#        listOfSections = listOfApicalSections + listOfBasalSections

listOfRandBgBasalLocsInSec = np.random.rand(len(listOfBgBasalSec))
listOfRandBgApicalLocsInSec = np.random.rand(len(listOfBgApicalSec))
listOfRandBasalLocationsInSection = np.random.rand(len(listOfRandBasalSectionInds))
listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
listOfRandInhLocationsInSection = float(2)/4 + 0.25*np.random.rand(len(listOfRandInhSectionInds))
#        listOfSegLocs = list(listOfRandApicalLocationsInSection) + list(listOfRandBasalLocationsInSection)

eventTime1 = []
eventTime2 = []
eventTime3 = []
eventTime4 = []
eventTime5 = []
eventTime6 = []

for k, section in enumerate(listOfBgApicalSec):
    eventTime1.append(silentTimeAtStart + np.random.uniform(0,simulationTime))#100*np.random.normal(0,1))
for k, section in enumerate(listOfBgBasalSec):
    eventTime2.append(silentTimeAtStart + np.random.uniform(0,simulationTime))#100*np.random.normal(0,1)) #simulationTime/2*np.random.rand(1)[0]
for k, section in enumerate(listOfApicalSections):
    eventTime3.append(silentTimeAtStart + delayTime + treeT*np.random.normal(0,1)) #gauss(0.5,0.2)
for k, section in enumerate(listOfBasalSections):
    eventTime4.append(silentTimeAtStart + delayTime + treeT*np.random.normal(0,1)) #simulationTime/2*np.random.rand(1)[0]
for k, section in enumerate(listOfInhSections):
    eventTime5.append(silentTimeAtStart + np.random.uniform(0,simulationTime)) #simulationTime/2*np.random.rand(1)[0]
    eventTime6.append(silentTimeAtStart + delayTime + treeT*np.random.normal(0,1)) #simulationTime/2*np.random.rand(1)[0]

    
#%% run the simulation
listOfSomaTraces = []               # all these needed only out of for(exp) loop
listOfDendTraces = [[]]*numA[-1]
listOfNexusTraces = []
spikes = []

for exp in range(numExp):
        
    if 1==0: print '0'#exp==numExp-2:
#        listOfRandInhSectionInds = []#[int(medSegment[exp])]*int(numApi)    
#        listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
#        listOfEvents = []
#        for k, section in enumerate(listOfBgApicalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgApicalLocsInSec[k]), eventTime1[k], 1, 1))
#        for k, section in enumerate(listOfBgBasalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgBasalLocsInSec[k]), eventTime2[k], 1, 1))
#        for k, section in enumerate(listOfBasalSections):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime4[k], 1, 1))
#
#    elif exp==numExp-3:
##        listOfRandInhSectionInds = []#[int(medSegment[exp])]*int(numApi)    
##        listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
#        listOfEvents = []
#        for k, section in enumerate(listOfBgApicalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgApicalLocsInSec[k]), eventTime1[k], 1, 1))
#        for k, section in enumerate(listOfBgBasalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgBasalLocsInSec[k]), eventTime2[k], 1, 1))
#        for k, section in enumerate(listOfApicalSections):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime3[k], 1, 1))
#
#    elif exp==numExp-1:
##        listOfRandInhSectionInds = []#[int(medSegment[exp])]*int(numApi)    
##        listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
#        for sec in h.allsec():
#            if hasattr(sec, 'gCa_HVAbar_Ca_HVA'):
#                sec.gCa_HVAbar_Ca_HVA = 0
#
#        listOfEvents = []
#        for k, section in enumerate(listOfBgApicalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgApicalLocsInSec[k]), eventTime1[k], 1, 1))
#        for k, section in enumerate(listOfBgBasalSec):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgBasalLocsInSec[k]), eventTime2[k], 1, 1))
#        for k, section in enumerate(listOfApicalSections):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime3[k], 1, 1))
#        for k, section in enumerate(listOfBasalSections):
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime4[k], 1, 1))

    else:
#        listOfRandInhSectionInds = [int(medSegment[exp])]*int(numApi)#randSecDist(cell.apic,dist[exp],20,numApi)# + np.random.randint(0,len(cell.apic)/partApi,numApi)
#        listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
        eventTime6 = []
        for k, section in enumerate(listOfInhSections):
#            eventTime5.append(silentTimeAtStart + np.random.uniform(0,simulationTime)) #simulationTime/2*np.random.rand(1)[0]
            eventTime6.append(silentTimeAtStart + delayTime + ApiBasInt[exp] + treeT*np.random.normal(0,1)) #simulationTime/2*np.random.rand(1)[0]
    ##        listOfBackgroundApicalSecInds = np.random.randint(0,len(cell.apic),int(numA[-1]*(1+np.random.normal(0,0.1))))#int(numApi))#
    #        listOfRandApicalSectionInds = randSecWeight(cell.apic,62,partApi,numA[exp])#medSeg + np.random.randint(-distance,distance,int(numApi))
    ##        listOfBgApicalSec = [cell.apic[x] for x in listOfBackgroundApicalSecInds]
    #        listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
    ##        listOfRandBgApicalLocsInSec = np.random.rand(len(listOfBgApicalSec))
    #        listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
        listOfEvents = []
        for k, section in enumerate(listOfBgApicalSec):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgApicalLocsInSec[k]), eventTime1[k], 1, 1))
        for k, section in enumerate(listOfBgBasalSec):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBgBasalLocsInSec[k]), eventTime2[k], 1, 1))
        for k, section in enumerate(listOfInhSections):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandInhLocationsInSection[k]), eventTime5[k], 1, 0))
        for k, section in enumerate(listOfApicalSections):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime3[k], 1, 1))
        for k, section in enumerate(listOfBasalSections):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime4[k], 1, 1))
        for k, section in enumerate(listOfInhSections):
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandInhLocationsInSection[k]), eventTime6[k], 1, 0))
        
    h.dt = 0.025
    recTime = h.Vector()
    recTime.record(h._ref_t)
    recVoltage = h.Vector()
    recVoltage.record(cell.soma[0](0.5)._ref_v)
    recDendVolt = [0]*numA[-1]
    for syn in range(numA[-1]):
        recDendVolt[syn] = h.Vector()
        recDendVolt[syn].record(cell.apic[60+syn/(numA[-1]/4)](float(syn%(numA[-1]/4))/(numA[-1]/4))._ref_cai)
    recNexusVolt = h.Vector()
    recNexusVolt.record(cell.apic[36](1)._ref_v)
    
    cvode.cache_efficient(1)
    h.finitialize(-80)#76)
    stopTime = totalSimDuration 
    neuron.run(stopTime)
    
    # plot the trace
    origRecordingTime = np.array(recTime.to_python()) # ugly workaround to recTime.as_numpy()
    origSomaVoltage = np.array(recVoltage.to_python()) # ugly workaround to recVoltage.as_numpy()
    origDendVoltage = [0]*numA[-1]
    for syn in range(numA[-1]):
        origDendVoltage[syn] = np.array(recDendVolt[syn].to_python()) # ugly workaround to recVoltage.as_numpy()
    origNexusVoltage = np.array(recNexusVolt.to_python()) # ugly workaround to recVoltage.as_numpy()
        
    recordingTime = np.arange(0,totalSimDuration,1.0/origNumSamplesPerMS)
    somaVoltage   = np.interp(recordingTime, origRecordingTime, origSomaVoltage)    
    dendVoltage = [0]*numA[-1]
    for syn in range(numA[-1]):
        dendVoltage[syn]   = np.interp(recordingTime, origRecordingTime, origDendVoltage[syn])    
    nexusVoltage   = np.interp(recordingTime, origRecordingTime, origNexusVoltage)    
    
    listOfSomaTraces.append(somaVoltage)
    for syn in range(numA[-1]):
        listOfDendTraces[syn].append(dendVoltage[syn])
    listOfNexusTraces.append(nexusVoltage)
    
#    origSpikes = []
#    
#    k = (silentTimeAtStart+delayTime-50)*origNumSamplesPerMS
#    while k < totalSimDuration*origNumSamplesPerMS:    # count spikes
#        if somaVoltage[k]>-20:
#            spikeTime = float(k)/origNumSamplesPerMS
#            numTemp = numSpikesPerExp[exp]
#            if numTemp>0 and spikeTime-origSpikes[numTemp-1]>15:
#                break
#            origSpikes.append(spikeTime)
#            numSpikesPerExp[exp] = numTemp + 1
#            numSpikes = numSpikes + 1
#            k = k+origNumSamplesPerMS*3
#        else:
#            k = k+5;
    
    #    spikes = []
    #    spikes.append(origSpikes)
ind+=1    
plt.figure('InhDt-CaI60 %s Traces inh:api:bas %s:20:%s in %s ms - #%s' 
           % (medSeg, numApi, numBas, treeT, ind) )
plt.title('Bas/api %s, api/%s inhLapi:bas %s:50:%s in %s ms, %s ms interval' 
          % (medSeg, partApi, numApi, numBas, treeT, ApiBasInt) )
plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
plt.axis(xmin=-50, xmax=100, ymin=-85, ymax=50)#min(somaVoltage)-5, ymax=max(somaVoltage)+5)
#xmax=stopTime-silentTimeAtStart-delayTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
tme = recordingTime-silentTimeAtStart-delayTime
i=0
for somaVoltageTrace,dendVoltageTrace,dend2VoltageTrace,nexusVoltageTrace in zip(listOfSomaTraces,listOfDendTraces[0],listOfDendTraces[-1],listOfNexusTraces):
    i+=1
    plt.subplot(5,2,i)#int('25'+str(i)))
    plt.title(i); plt.axis(xmin=-10, xmax=90, ymin=-85, ymax=65)#100+(i-1)*200
    if i>5: plt.xlabel('Time [ms]') 
    if i%5==1: plt.ylabel('Voltage [mV]') #
    plt.plot(tme, somaVoltageTrace, tme, dendVoltageTrace*1e5, tme, dend2VoltageTrace, tme, nexusVoltageTrace)
    if i==10: break
#%% measure time over each threshold (or Omega) to get final synaptic weight
ind+=1
th_p = np.linspace(.5e-4,1e-3,num=20)
th_d = th_p*0.6
depDur = [[0 for i in range(numA[-1])] for j in range(len(th_p))]
potDur = [[0 for i in range(numA[-1])] for j in range(len(th_p))]
w = [[1 for i in range(numA[-1])] for j in range(len(th_p))]

lines = [0 for x in range(len(th_p))]#len(clusterSize))]#totalSyn))]
colors = cm.get_cmap('jet')#('YlGnBu')#jet')#('cool')#
plt.figure('final synaptic weights by relative time over thresholds %s' %(ind))
plt.title('final synaptic weights')

for j in range(len(th_p)):
#    w = [1]*numA[-1]
    for syn in range(numA[-1]):
        depDur[j][syn] = np.sum([(listOfDendTraces[syn][syn][i]<th_p[j])*(listOfDendTraces[syn][syn][i]>th_d[j]) for i in range(12000)])
        potDur[j][syn] = np.sum([listOfDendTraces[syn][syn][i]>th_p[j] for i in range(12000)])
        w[j][syn] = w[j][syn] - depDur[j][syn]/float(1200) + potDur[j][syn]/float(400)
    lines[j]=plt.plot(np.linspace(60,63.8,num=numA[-1]),w[j],'.',label='%s' % int(th_p[int(j)]),color=colors(float(20-j)/20))
plt.legend()#bbox_to_anchor=(0.9,0.6)) #lines,['2.26','3.2','4.53','6.4','9.05'])

#plt.plot(np.linspace(60,63.8,num=numA[-1]),w,'.')
plt.figure('final synaptic weights by relative time over thresholds %s' %(ind))
#%% plot same with color
ind+=1
colors = cm.get_cmap('jet')
plt.figure('w(meds=%s) #%s' %(medSeg,ind))

for i in range(numA[-1]):
    plt.plot(60+i*4./numA[-1],w[i],'.',color=colors((np.log2(w[i])+2)/4))#0.75)/1.75))
#plt.plot(np.linspace(12,23,num=12),spksdt0[16:],'.k')
plt.xlabel('Proximal -> Distal'); plt.ylabel('w')#Spikes')
plt.figure('w(meds=%s) #%s' %(medSeg,ind))
plt.savefig('w_meds61_bas100api20num10_th25_4')

#%% w -> log2w
log2w = [[0 for i in range(numA[-1])] for j in range(len(th_p))]
for j in range(len(th_p)):
    for i in range(numA[-1]):
        log2w[j][i] = np.log2(w[j][i])

#%% current clamp exp

#simulationTime    = 400
#silentTimeAtStart = 100
#delayTime = 200
##    ApicalBasalInterval = 60
##    treeTime = 20
#silentTimeAtEnd   = 100
#
#origNumSamplesPerMS = 20
#totalSimDuration = simulationTime + silentTimeAtStart + silentTimeAtEnd
#    
#listOfSomaTraces = []
#spikes = []
#numSpikes = 0
#numSpikesPerExp = [0]*numExp
numExp=1
medSeg=0
ampSom=1.8

def runIClampSim(cell,ApiBasInt,treeT,numBas,numApi,partApi,medSeg,ampSom,numExp):
    
    simulationTime    = 400
    silentTimeAtStart = 100
    delayTime = 200
#    ApicalBasalInterval = 60
#    treeTime = 20
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
    #        medS = medSeg
        
        listOfEvents = []
        #eventTime = silentTimeAtStart #+ simulationTime/2 + simulationTime/2*np.random.rand(1)[0]
        #listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(L5PC.dend[1](0.5), eventTime, 2))
        listOfRandBasalSectionInds  = np.random.randint(0,len(cell.dend),int(numBas))
        #listOfRandApicalSectionInds = []
        listOfRandApicalSectionInds = medSeg + np.random.randint(0,len(cell.apic)/partApi,int(numApi))
        #for clust in range(int(partApi)):
        #    np.append(listOfRandApicalSectionInds, np.random.randint(0,len(cell.apic),1) 
        #                                    + np.random.randint(0,len(cell.apic)/partApi,int(numApi/partApi)), axis=0)
    #            medS = medS + 128/partApi
        listOfBasalSections  = [cell.dend[x] for x in listOfRandBasalSectionInds]
        listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
        #listOfApicalSections = []
        
    #        listOfSections = listOfApicalSections + listOfBasalSections
        
        listOfRandBasalLocationsInSection = np.random.rand(len(listOfRandBasalSectionInds))
        listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
    #        listOfSegLocs = list(listOfRandApicalLocationsInSection) + list(listOfRandBasalLocationsInSection)
        #listOfSegLocs = list(listOfRandBasalLocationsInSection)
        
        listOfEvents = []
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + 50*np.random.normal(0,1)#1,0.2)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 2, 1))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + 50*np.random.normal(0,1)#1,0.2) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 2, 1))
            
#        h("access L5PC.soma[0]")
#        h("objref basalIClamp")
#        h.basalIClamp = h.IClamp(0.5)
#        h("basalIClamp.del = 300") #silentTimeAtStart+delayTime+ApiBasInt")
#        h("basalIClamp.dur = 5")
#        ampCmd = "basalIClamp.amp = %s" % (ampSom)
#        h(ampCmd)
            
        basalIClamp = h.IClamp(cell.soma[0](0.5))
        basalIClamp.delay = 300
        basalIClamp.dur = 5
        basalIClamp.amp = ampSom#1 #20
                    
        ##%% run the simulation
        recTime = h.Vector()
        recTime.record(h._ref_t)
        recVoltage = h.Vector()
        recVoltage.record(cell.soma[0](0.5)._ref_v)
        recNexusVolt = h.Vector()
        recNexusVolt.record(cell.apic[36](1)._ref_v)
        
        cvode.cache_efficient(1)
        h.finitialize(-76)
        stopTime = totalSimDuration
        neuron.run(stopTime)
        
        # plot the trace
        origRecordingTime = np.array(recTime.to_python()) # ugly workaround to recTime.as_numpy()
        origSomaVoltage = np.array(recVoltage.to_python()) # ugly workaround to recVoltage.as_numpy()
        origNexusVoltage = np.array(recNexusVolt.to_python()) # ugly workaround to recVoltage.as_numpy()
    
        recordingTime = np.arange(0,totalSimDuration,1.0/origNumSamplesPerMS)
        somaVoltage   = np.interp(recordingTime, origRecordingTime, origSomaVoltage)    
        nexusVoltage   = np.interp(recordingTime, origRecordingTime, origNexusVoltage)    
    
        listOfSomaTraces.append(somaVoltage)
#        listOfNexusTraces.append(nexusVoltage)
            
        origSpikes = []
        tempSpikes = 0
        
        k = (silentTimeAtStart+delayTime-50)*origNumSamplesPerMS #int(np.min([0,ApiBasInt]))
        while k < (totalSimDuration-silentTimeAtEnd)*origNumSamplesPerMS:
            if somaVoltage[k]>-10:
                tempTime = float(k)/origNumSamplesPerMS
#                if tempSpikes==1 and tempTime-origSpikes[-1]>20:
#                    tempSpikes = 0
#                    numSpikes -= 1
#                    del origSpikes[-1]
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
        
        ind+=1
        plt.figure('I-clamp exp %snA %sms %s biophys %s' % (basalIClamp.amp,basalIClamp.dur,ind,biophysicalModelFilename))
        plt.plot(recordingTime, somaVoltage, recordingTime, nexusVoltage)
        plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
        plt.axis(xmin=281, xmax=359, ymin=-89, ymax=42)
        
        basalIClamp.delay = 0
        basalIClamp.dur = 0
        basalIClamp.amp = 0
        
    if (experiment+1)%5==0 or (time.time()-startTime)/60>5 or numExp<5: 
            print "Interval %s somatic amp %s exp. # %s took %.3f minutes" % (ApiBasInt,ampSom,experiment+1, (time.time()-startTime)/60)
        
    print "Mean no. of spikes: %s FR: %s" % (float(numSpikes)/numExp,np.mean(freq))
    return float(numSpikes)/numExp,np.mean(freq)#recordingTime,somaVoltage,nexusVoltage #float(numSpikes)/numExp

#%%
plt.figure()
plt.title('%s traces, api/ %s api:bas %s : %s in %s ms, %s ms interval' 
          % (numExp, partApi, numApi, numBas, treeT, ApiBasInt) )
plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
plt.axis(xmin=0, xmax=stopTime-silentTimeAtStart-delayTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
for somaVoltageTrace in listOfSomaTraces:
    plt.plot(recordingTime-silentTimeAtStart-delayTime, somaVoltageTrace)    
        
#%% run simulation on some parameter pair, plot the space

ApicalBasalInterval = 0 #np.linspace(-20,20,num=9) #ApicalBasalInterval = np.linspace(-30,40,num=36)
#totSyn = np.linspace(0,500,num=21)
numBasal = 0 #np.linspace(0,400,num=17)
numApical = 0 #400-numBasal
partApical = 10 #np.logspace(1,7,num=13,base=2)
medSegment = 20 #np.linspace(0,100,num=21)
treeTime = 20 #np.linspace(5,100,num=20)
ampApi = np.linspace(0,5,num=11)
numExperiments = 1

amp = np.linspace(0,10,num=101)
spks = [0 for i in range(len(amp))] #for j in range(len(ampApi))]
frqs = [0 for i in range(len(amp))] #for j in range(len(ampApi))]
trc = [[] for i in range(len(amp))]

#i = 0
j = 0

start = time.time()

#for ApiBasInd in ApicalBasalInterval:
#    print "Running for interval: %s [ms]" % (int(ApiBasInd))
for ampS in amp:
#        if partA>110:
#            break
#        numA = 0.75*totS
#        print "Running for 1/", partA, "of apical tree"
    print "Running for ", ampS, "somatic amplitude"
    spks[j],frqs[j] = runIClampSim(L5PC,ApicalBasalInterval,treeTime,numBasal,numApical,partApical,medSegment,ampS,numExperiments)
    j = j+1
#j = 0
#i = i+1
    
print "Total running time was: ", (time.time()-start)/3600, "hours"

#%% plot data in 3D

percent = 100/partApical

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title('Spikes as function of $\Delta$t and apical current amplitude') 
#apical segment size (by denominator of entire tree)') # % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
plt.xlabel('$\Delta$t [ms]'); plt.ylabel('apical current amplitude (nA)') #'Apical segment size (denom.)')
#ax.set_zlabel('Mean # of spikes'); 
ax.set_zlim(0,4)
plt.axis(xmin=min(ApicalBasalInterval), xmax=max(ApicalBasalInterval), 
         ymin=min(ampApi), ymax=max(ampApi))
apbas4plt = [[ind for ind in ApicalBasalInterval] for j in range(len(ampApi))]
numap4plt = [[j for ind in range(len(ApicalBasalInterval))] for j in ampApi]
surf = ax.plot_surface(apbas4plt,numap4plt,np.asarray(spks),cmap=cm.coolwarm,vmin=0,vmax=4)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#%% cut down spks

spksmeancut = [[0 for i in range(len(ApicalBasalInterval))] for j in range(len(medSegment))]#partApical[:10]))]

for i in range(len(ApicalBasalInterval)):#numApical[:-10])):#
    for j in range(len(medSegment)):#partApical[:10])):
        spksmeancut[j][i] = spksmean[j+4][i]
#        frqs2[i][j] = frqs[i][j]
        
#%% temp: run 1 line
    
dt = np.linspace(-50,50,num=51)
spks1 = [0 for i in range(len(dt))]
    
j = 0

for i in range(len(ApicalBasalInterval)):
    for j in range(len(medSegment)-4):#partApical[:10])):
        spksmeancut1[j][i] = spksmean14[j+4][i]

#    print "Running for interval: %s [ms]" % (int(ApiBasInd))
#    for ampA in ampApi:
#        if partA>110:
#            break
#        numA = 0.75*totS
#        print "Running for 1/", partA, "of apical tree"
#        print "Running for ", ampA, "apical amplitude"
    spks1[j] = runSim(L5PC,i,10,250,150,10,50,10)
    j = j+1
        
# print "Total running time was: ", (time.time()-start)/3600, "hours"

#%% plot single line

fig = plt.figure()
plt.title('Spikes as function of $\Delta$t (150 apical + 250 basal synapses)')
plt.xlabel('$\Delta$t [ms]'); plt.ylabel('Mean # of spikes') #'Apical segment size (denom.)')
plt.axis(xmin=min(dt), xmax=max(dt), 
         ymin=0, ymax=4)
plt.plot(dt, spks1)
        
#%% fit line with function

#%% stitch 2 spks arrays interspersed:
ApicalBasalInterval = np.linspace(-40,60,num=51) #[x for xs in a for x in xs]
spksnew4 = [[0 for i in range(len(spksnew2[0]))] for j in range(7)]#ApicalBasalInterval))]

for i in range(len(ApicalBasalInterval)):
    for j in range(len(treeTime)):
        if i%2:
            spksnew[j][i] = spks0[j][i/2]
        else:
            spksnew[j][i] = spks[j][i/2]

#%% stitch 2 spks arrays:
for i in range(len(spks2[0])):
    for j in range(len(spks2)):
        spks1[j][i] = spksmean1[j][i]

#%% stitch intermingled tot_api
spksmean = [[0 for x in range(len(spksmean0[0])+len(spksmean1[0]))] 
                    for y in range(len(spksmean0))]

for j in range(len(spksmean)):
    for i in range(2):#len(spksnew2)):
        spksmean[j][i] = spksmean0[j][i]
    spksmean[j][2] = spksmean1[j][0]
    spksmean[j][3] = spksmean0[j][2]
    for k in range(3):
        spksmean[j][k+4] = spksmean1[j][k+1]
    spksmean[j][7] = spksmean0[j][3]

#%% array to list
def switchxy(arr):
    arrnew = [[0 for i in range(len(arr))] for j in range(len(arr[0]))]
    for i in range(len(arr[0])):
        for j in range(len(arr)):
            arrnew[i][j]=arr[j][i]
    return arrnew

#%% take 10 spks arrays and make one mean
#spksall = []
frqsmean2 = [[0 for x in range(len(bas300api_dt0pt10rnd_spks0[0]))] 
                    for y in range(len(bas300api_dt0pt10rnd_spks0))]
spksmean2 = [[0 for x in range(len(bas300api_dt0pt10rnd_spks0[0]))] 
                    for y in range(len(bas300api_dt0pt10rnd_spks0))]
for i in range(len(bas300api_dt0pt10rnd_spks0)):
    for j in range(len(bas300api_dt0pt10rnd_spks0[0])):
        frqsmean2[i][j] = float(1000)/10*(bas300api_dt0pt10rnd_frqs0[i][j] + 
                        bas300api_dt0pt10rnd_frqs1[i][j] + 
                        bas300api_dt0pt10rnd_frqs2[i][j] + 
                        bas300api_dt0pt10rnd_frqs3[i][j] + 
                        bas300api_dt0pt10rnd_frqs4[i][j] + 
                        bas300api_dt0pt10rnd_frqs5[i][j] + 
                        bas300api_dt0pt10rnd_frqs6[i][j] + 
                        bas300api_dt0pt10rnd_frqs7[i][j] + 
                        bas300api_dt0pt10rnd_frqs8[i][j] +
                        bas300api_dt0pt10rnd_frqs9[i][j])
        spksmean2[i][j] = float(1)/10*(bas300api_dt0pt10rnd_spks0[i][j] + 
                        bas300api_dt0pt10rnd_spks1[i][j] + 
                        bas300api_dt0pt10rnd_spks2[i][j] + 
                        bas300api_dt0pt10rnd_spks3[i][j] + 
                        bas300api_dt0pt10rnd_spks4[i][j] + 
                        bas300api_dt0pt10rnd_spks5[i][j] + 
                        bas300api_dt0pt10rnd_spks6[i][j] + 
                        bas300api_dt0pt10rnd_spks7[i][j] + 
                        bas300api_dt0pt10rnd_spks8[i][j] +
                        bas300api_dt0pt10rnd_spks9[i][j])
#        if frqsmean2[i][j]==0: frqsmean2[i][j]=1e-3
        
#%% plot inh prox-dist with color
colors = cm.get_cmap('jet')
plt.figure('inh colorspk2')

for i in range(16):
    plt.plot(i+1,spksdt0[i+4],'.',color=colors((float(spksdt0[i+4])-0.75)/1.75))
#plt.plot(np.linspace(12,23,num=12),spksdt0[16:],'.k')
plt.xlabel('Proximal -> Distal'); plt.ylabel('Spikes')
        
#%% plot 2d density heatmap of spks-freq
#from scipy.stats import kde
from scipy.stats.kde import gaussian_kde

x=spks; y=frqsk
Z, xedges, yedges = np.histogram2d(x,y)
k = gaussian_kde(np.vstack([x,y]))
xi, yi = np.mgrid[min(x):max(x):len(x)**0.5*1j,min(y):max(y):len(y)**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.figure('pcolormesh')
#plt.pcolormesh(xedges, yedges, Z.T)
plt.pcolormesh(xi, yi, zi.reshape(xi.shape))#, alpha=0.5)

#%% create mean and rmse of spks, plot for x

spksAB4plot = [[0 for i in range(50)] for j in rangchecke(len(medSegment))]
meanspks = [0 for i in range(len(medSegment))]
stdspks = [0 for i in range(len(medSegment))]

for i in range(len(medSegment)):
    for j in range(50):
        spksAB4plot[i][j] = spksAB[j][i]
    meanspks[i] = np.mean(spksAB4plot[i])
    stdspks[i] = np.std(spksAB4plot[i])
    
#meanspks = mean(spksAB[0])

SecDistance = [0 for i in range(len(medSegment))]
h.distance(sec=L5PC.soma[0])
realMedSeg = 5 #len(L5PC.apic)/partApical/2
partApical = 10

for i in range(len(medSegment)):
    if i==len(medSegment)-1:
        medS = len(L5PC.apic)
    else:
        medS = medSegment[i]+len(L5PC.apic)/partApical
    allSecDist = []
    for j in range(int(medSegment[i]),int(medS)):
        allSecDist.append(h.distance(0,sec=L5PC.apic[j]))
    SecDistance[i] = np.mean(allSecDist) #h.distance(0,sec=L5PC.apic[int(medSegment[i]+realMedSeg)])

sortedSpksAB = [x for _,x in sorted(zip(SecDistance,meanspks))]    
sortedStdsAB = [x for _,x in sorted(zip(SecDistance,stdspks))]    
sortSecDist = sorted(SecDistance)

fig = plt.figure()
plt.plot(sortSecDist,sortedSpksA,'.',sortSecDist,sortedSpksAB,'.',sortSecDist,sortedSpks,'.')
plt.fill_between(sortSecDist,np.subtract(sortedSpks,sortedStds),[x+y for (x,y) in zip(sortedSpks,sortedStds)],alpha=0.5,facecolor='g')
plt.fill_between(sortSecDist,np.subtract(sortedSpksA,sortedStdsA),[x+y for (x,y) in zip(sortedSpksA,sortedStdsA)],alpha=0.5,facecolor='c') 
plt.fill_between(sortSecDist,np.subtract(sortedSpksAB,sortedStdsAB),[x+y for (x,y) in zip(sortedSpksAB,sortedStdsAB)],alpha=0.5,facecolor='y')
#plt.show()
plt.title('Spikes to distance of inhibition from soma')
plt.xlabel('Mean distance of IPSPs from soma [$\mu$m]'); plt.ylabel('Mean spikes')
plt.legend(labels=['GABA_A','GABA_A+B','Control'],loc=4)

#%% plot spks-dt line for fitting with function

#spks4plot = [0 for i in range(len(ApicalBasalInterval))] #for j in range(len(medSegment))]
meanspks = [[ind for ind in range(len(ApicalBasalInterval))] for j in range(len(treeTime))]
#stdspks = [0 for i in range(len(medSegment))]

for i in range(len(ApicalBasalInterval)):
    for j in range(len(treeTime)):
#    for j in range(50):
#        spksAB4plot[i][j] = spksAB[j][i]
        meanspks[j][i] = spks[j][i][0]#np.mean(spks[i])
#    stdspks[i] = np.std(spksAB4plot[i])
    
#meanspks = mean(spksAB[0])
#%%

fig = plt.figure()
plt.plot(dt,spks1,'.') #,sortSecDist,sortedSpksAB,'.',sortSecDist,sortedSpks,'.')
#plt.show()
plt.title('Spikes to ApicalBasalInterval')
plt.xlabel('ApicalBasalInterval'); plt.ylabel('Mean spikes')
#plt.legend(labels=['GABA_A','GABA_A+B','Control'],loc=4)

#%% define function and fit

from scipy import optimize

treeT = 1

def func(x,baseline,amp,width,expbias,expdenom):
    return baseline+amp*(np.sign(width-abs(x+treeT))+1)*np.exp(abs((x-expbias)/expdenom))

params, paramscov = optimize.curve_fit(func,ApicalBasalInterval,spks,p0=[2,1,20,5,10])

print params

res = spks-func(ApicalBasalInterval,params[0],params[1],
                    params[2],params[3],params[4])
ss_res = np.sum(res**2)
ss_tot = np.sum((spks-np.mean(spks))**2)
r_squared = 1-(ss_res/ss_tot)

print r_squared

fig = plt.figure()
plt.plot(ApicalBasalInterval,spks,'.',
         ApicalBasalInterval,func(ApicalBasalInterval,params[0],params[1],
                                  params[2],params[3],params[4])) 
            #,sortSecDist,sortedSpksAB,'.',sortSecDist,sortedSpks,'.')
#plt.show()
plt.title('Spikes to $\Delta$t')
plt.xlabel('$\Delta$t [ms]'); plt.ylabel('Mean spikes')
plt.legend(labels=['Measured','Fitted function'],loc='best')
plt.text(max(ApicalBasalInterval)-20,max(spks)-.5,'R^2=%s' % (round(r_squared,3)))
    