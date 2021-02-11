import os
import sys
#sys.path.append('C:\nrn\lib\python')
os.chdir('/ems/elsc-labs/segev-i/eilam.goldenberg/Documents/Active Cell Real Morphology/')

#os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ';C:\Python27' + ';C:\nrn' + ';C:\nrn\lib\hoc\import3d' + ';C:\nrn\lib\python'
#os.environ['NEURONHOME'] = 'C:\nrn'
#os.environ['Path'] = 'C:\nrn;C:\nrn\bin'

from neuron import h
from neuron import gui

#%%

#import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
#from random import shuffle
#from random import randint
#from random import gauss
import time
import math
#from vector import vector, plot_peaks
#from libs import detect_peaks

#%%

from sklearn import decomposition
from sklearn import cluster
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation

#%%

h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")

cvode = h.CVode()
cvode.active(0)

morphologyFilename = "morphologies/cell1.asc"
#morphologyFilename = "morphologies/cell2.asc"
#morphologyFilename = "morphologies/cell3.asc"
#morphologyFilename = "morphologies/V1.ASC"

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
for sec in h.allsec():
    if hasattr(sec, 'gCa_HVAbar_Ca_HVA'):
        sec.gCa_HVAbar_Ca_HVA = 0
    

#%% inspect the created shape

shapeWindow = h.PlotShape()
shapeWindow.exec_menu('Show Diam')

#%% helper functions 

def Add_NMDA_SingleSynapticEventToSegment(segment, activationTime, synapseWeight, exc_inh):

#    synapse = h.ProbAMPANMDA2(segment)
#    synapse = h.ProbAMPANMDA_EMS(segLoc,sec=section)
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
#        synapse.Use = 1
#        synapse.u0 = 0
#        synapse.Dep = 0
#        synapse.Fac = 0

    else:           # excitatory
        synapse = h.ProbAMPANMDA2(segment)
#    synapse = h.ProbAMPANMDA_EMS(segLoc,sec=section)
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

# GABA A+B synapse
def Add_GABA_SingleSynapticEventToSegment(segment, activationTime, synapseWeight): #DefineSynapse_GABA_AB(segment, gMax = 0.001):
    synapse = h.ProbGABAAB_EMS(segment)

#    synapse.GABAB_ratio = 0.0
    synapse.Use = 1.0
#    synapse.u0 = 0
    synapse.Dep = 0
    synapse.Fac = 0

    netStimulation = h.NetStim()                   
    netStimulation.number = 1
    netStimulation.start = activationTime
    
    netConnection = h.NetCon(netStimulation,synapse)
    netConnection.delay = 0
    netConnection.weight[0] = synapseWeight

    return netStimulation,netConnection,synapse


#%% add some random NMDA synapses and plot a somatic trace just to see all things are alive and kicking

# simulationTime    = 400
# silentTimeAtStart = 100
# delayTime = 200
ApicalBasalInterval = 60
treeTime = 20
# silentTimeAtEnd   = 100


# origNumSamplesPerMS = 20
# totalSimDuration = simulationTime + silentTimeAtStart + silentTimeAtEnd

numBasal = 50
numApical = 20
partOfApical = 10
medSegment = 0

numExperiments = 5

def runSim(cell,ApiBasInt,treeT,numBas,numApi,partApi,medSeg,numExp):
    
    simulationTime    = 400
    silentTimeAtStart = 100
    delayTime = 200
#    ApicalBasalInterval = 60
#    treeTime = 20
    silentTimeAtEnd   = 100
#    numBasal
    origNumSamplesPerMS = 40 # was 20!!!
    totalSimDuration = simulationTime + silentTimeAtStart + silentTimeAtEnd
        
    listOfSomaTraces = []
    spikes = []
    numSpikes = 0
#    numSpikesPerExp = [0]*numExp
    
    for experiment in range(numExp):
    
        startTime = time.time()
        
        
        listOfRandBasalSectionInds  = np.random.randint(0,len(cell.dend),int(numBas))
        listOfRandApicalSectionInds = 60 + np.random.randint(0,len(cell.apic)/partApi,int(40))#numApi))
        listOfRandInhSectionInds = int(medSeg) + np.random.randint(0,len(cell.apic)/partApi,numApi) #int(numInh))
#        listOfRandObliqueSectionInds = np.random.randint(0,len(cell.apic)/partApi,int(40-numApi)) #obliques
        listOfBasalSections  = [cell.dend[x] for x in listOfRandBasalSectionInds]
        listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
        listOfInhSections = [cell.apic[x] for x in listOfRandInhSectionInds]
#        listOfObliqueSections = [cell.apic[x] for x in listOfRandObliqueSectionInds]
        #listOfApicalSections = []
        
#        listOfSections = listOfApicalSections + listOfBasalSections
        
        listOfRandBasalLocationsInSection = np.random.rand(len(listOfRandBasalSectionInds))
        listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
        listOfRandInhLocationsInSection = np.random.rand(len(listOfRandInhSectionInds))
#        listOfRandObliqueLocationsInSection = np.random.rand(len(listOfRandObliqueSectionInds))
#        listOfSegLocs = list(listOfRandApicalLocationsInSection) + list(listOfRandBasalLocationsInSection)
        #listOfSegLocs = list(listOfRandBasalLocationsInSection)
        
        listOfEvents = []
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 2, 1))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 2, 1))
        
        for k, section in enumerate(listOfInhSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandInhLocationsInSection[k]), eventTime, 2, 0))
        
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + delayTime + 5*np.random.normal(0,1) #gauss(0.5,0.2)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 2, 1))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + delayTime + 20 + 5*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 2, 1))
        
        for k, section in enumerate(listOfInhSections):
            eventTime = silentTimeAtStart + delayTime + ApiBasInt + treeT*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandInhLocationsInSection[k]), eventTime, 2, 0))

        #add obliques
#        for k, section in enumerate(listOfObliqueSections):
#            eventTime = silentTimeAtStart + delayTime + treeT*np.random.normal(1,0.2) #simulationTime/2*np.random.rand(1)[0]
#            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandObliqueLocationsInSection[k]), eventTime, 2))
        
        #listOfEvents = []
        #for k, section in enumerate(listOfSections):
        #    eventTime = silentTimeAtS10tart + simulationTime*np.random.rand(1)[0]
        #    listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfSegLocs[k]), eventTime, 2))
        1
        #eventTime = silentTimeAtStart #+ simulationTime/2 + simulationTime/2*np.random.rand(1)[0]
        #listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(L5PC.dend[1](0.5), eventTime, 2))
        
        ##%% run the simulation
        h.dt = 0.025
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
    
        k = (silentTimeAtStart+delayTime-50)*origNumSamplesPerMS
        while k < (totalSimDuration-silentTimeAtEnd)*origNumSamplesPerMS:
            if somaVoltage[k]>-20:
                tempTime = float(k)/origNumSamplesPerMS
                if tempSpikes > 0 and tempTime-origSpikes[-1]>20:
                    break
                origSpikes.append(tempTime)
                # numSpikesPerExp[experiment] = tempSpikes + 1
                numSpikes = numSpikes + 1
                tempSpikes = 1 # numSpikesPerExp[experiment]
                k = k+origNumSamplesPerMS*3
            else:
                k = k+5 # was 1 before
        
    #    spikes = []
        spikes.append(origSpikes)
        
        
    #    plt.figure()
    #    plt.plot(recordingTime, somaVoltage)
    #    plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
    #    plt.axis(xmin=0, xmax=stopTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
        
        #listOfEvents = []
        if (experiment+1)%5==0 or (time.time()-startTime)/60>5 or numExp<5: 
            print "medSeg %s dt %s exp. # %s took %.3f minutes" % (medSeg,ApiBasInt,experiment+1, (time.time()-startTime)/60)
#        elif (time.time()-startTime)/60>5:
#            print "Exp. %s took %.3f minutes" % (experiment+1, (time.time()-startTime)/60)
#        elif numExp<5:
#            print "Exp. %s took %.3f minutes" % (experiment+1, (time.time()-startTime)/60)
        
    print "Mean no. of spikes: %s" % (float(numSpikes)/numExp)
    return float(numSpikes)/numExp, listOfSomaTraces, recordingTime

#%% run simulation on some parameter pair, plot the space

ApicalBasalInterval = 0 #np.linspace(-20,40,num=31) #36)
numBasal = np.linspace(0,200,num=21) #33)
numApical = 200-numBasal #np.linspace(20,40,num=21)
numOblique = 40-numApical
#numI = 50
#totalSyn = np.linspace(0,500,num=21)
partApical = 10 #np.logspace(0,7,num=15,base=2)
medSegment = np.linspace(0,100,num=21)
#medSegment = 50
ApiBasInd = ApicalBasalInterval
treeTime = np.logspace(0,7,num=15,base=2)

spks = [[0 for i in range(len(medSegment))] for j in range(len(treeTime))]
#trc = spks
#tme = trc

i = 0
j = 0

start = time.time()

#for ApiBasInd in ApicalBasalInterval:
for medS in medSegment:
#    print "Running for interval: %s [ms]" % (int(ApiBasInd))
#    for totalS in totalSyn:
#    for medS in medSegment:
    if medS==100:
        partApical = 11
    for treeT in treeTime:
#        print "Running for apical # %s" % (int(numA))
#        numA = int(totalS*0.4)
        spks[j][i],_,_ = runSim(L5PC,ApiBasInd,treeT,60,100,partApical,medS,10) #0syn ctrl
        j = j+1
    j = 0
    i = i+1
    
print "Total running time was: ", (time.time()-start)/3600, "hours"

#%% plot data in 3D
indx+=1
fig = plt.figure('Spikes by inh dt and prox-dist, 80excmeds62dt0sd10#20 # %s' % (indx)) 
ax = fig.gca(projection='3d')
plt.title('Spikes by inh $\Delta$t & proximal-distal from exc') #syn # and  % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
plt.xlabel('$\Delta$t to inhibition [ms]'); plt.ylabel('proximal-distal')## of inhibitory synapses')#
plt.axis(xmin=min(ApicalBasalInterval), xmax=max(ApicalBasalInterval))#, ymin=min(medSegment), ymax=max(medSegment))
#ax.set_zlabel('Mean no. of spikes')
apbas4plt = [[ind for ind in ApicalBasalInterval] for j in range(len(dist))]#medSegment))]
numap4plt = [[j for ind in range(len(ApicalBasalInterval))] for j in dist]#medSegment]
surf = ax.plot_surface(apbas4plt,numap4plt,np.asarray(spksmean03),cmap=cm.jet)#,vmin=0,vmax=4)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

#%%

plt.figure()
plt.title('%s traces, api/ %s api:bas %s : %s in %s ms, %s ms interval' 
          % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
plt.axis(xmin=0, xmax=stopTime-silentTimeAtStart-delayTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
for somaVoltageTrace in listOfSomaTraces:
    plt.plot(recordingTime-silentTimeAtStart-delayTime, somaVoltageTrace)
#%% recalculate spikes
spikes = []
numSpikes = 0
experiment = 0
numSpikesPerExp = [0]*numExperiments

for somaVoltageTrace in listOfSomaTraces:
    origSpikes = []
    k = (silentTimeAtStart+delayTime)*origNumSamplesPerMS
    while k < (totalSimDuration-220)*origNumSamplesPerMS:
        if somaVoltageTrace[k]>0:
            origSpikes.append(float(k)/origNumSamplesPerMS)
            numSpikesPerExp[experiment] = numSpikesPerExp[experiment] + 1
            numSpikes = numSpikes + 1
            k = k+origNumSamplesPerMS*3
        else:
            k = k+1;
    spikes.append(origSpikes)
    experiment = experiment + 1

#%% raster plot spikes
spikesMean = float(numSpikes)/numExperiments
spikesStd = np.std(numSpikesPerExp)//0.01*1e-2

plt.figure()
plt.title('%s trials, api/ %s ap:bas %s : %s in %s ms, %s ms inter; Mean %s Std. %s' % (numExperiments,partOfApical,numApical,numBasal,treeTime,ApicalBasalInterval,spikesMean,spikesStd))
plt.xlabel('Time [ms]'); plt.ylabel('trial')
plt.axis(xmin=min(min(spikes)), xmax=stopTime-100)
i = 0 #zeros(length(spikes))
for k in spikes:
    i = i + 1
    plt.plot(k, [-i]*len(spikes[i-1]), 'k.')

#%% count mean spikes
#lines = spikes.split("\n")

#for line in spikes:
#    lines.append( line.split(",") )

#%%

print "Mean number of spikes: ", float(numSpikes)/numExperiments
print "Std: ", np.std(numSpikesPerExp)
    

#%%    spikes.append(detect_peaks(somaVoltageTrace, mph=-60, mpd=origNumSamplesPerMS))

plt.figure()
for ind in spikes:
    plt.plot(recordingTime[ind-origNumSamplesPerMS*50:ind], listOfSomaTraces[ind-origNumSamplesPerMS*50:ind],'*')
    
    