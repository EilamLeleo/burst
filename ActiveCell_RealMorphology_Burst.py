import os
import sys
os.chdir('C:/Users/Leleo/Documents/Active Cell Real Morphology/')

from neuron import h
from neuron import gui

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import math

sk = False

if sk==True:
    from sklearn import decomposition
    from sklearn import cluster
    from sklearn import linear_model
    from sklearn import ensemble
    from sklearn import cross_validation

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

biophysicalModelTemplateFilename = "L5PCtemplate.hoc"
#biophysicalModelTemplateFilename = "L5PCtemplate_2.hoc"

h.load_file(biophysicalModelFilename)
h.load_file(biophysicalModelTemplateFilename)
L5PC = h.L5PCtemplate(morphologyFilename)

#%% set dendritic VDCC g=0

VDCC_g = 1

if VDCC_g==0:
    for sec in h.allsec():
#        if hasattr(sec, 'gCa_LVAstbar_Ca_LVAst'):
#            sec.gCa_LVAstbar_Ca_LVAst = 0
        if hasattr(sec, 'gCa_HVAbar_Ca_HVA'):
            sec.gCa_HVAbar_Ca_HVA = 0 

#%% inspect the created shape

shapeWindow = h.PlotShape()
shapeWindow.exec_menu('Show Diam')

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
    isis = []
    numSpikes = 0
    numSpikesPerExp = [0]*numExp
    freq = [0]*numExp
    
    for experiment in range(numExp):
    
        startTime = time.time()
                
        listOfRandBasalSectionInds  = np.random.randint(0,len(cell.dend),int(numBas))
        distance = math.ceil(len(cell.apic)/float(partApi)/2)
        listOfRandApicalSectionInds = medSeg + np.random.randint(-distance,distance,int(numApi))
        # make anything over apical size be deduced by that value
        if int(numApi)>0 and max(listOfRandApicalSectionInds) > len(cell.apic)-1:
            deduce = max(listOfRandApicalSectionInds) - len(cell.apic)+1
            for x in range(numApi):
                listOfRandApicalSectionInds[x] -= deduce
#        listOfRandObliqueSectionInds = np.random.randint(0,len(cell.apic)/partApi,0)#int(40-numApi)) #obliques
        listOfBasalSections  = [cell.dend[x] for x in listOfRandBasalSectionInds]
        listOfApicalSections = [cell.apic[x] for x in listOfRandApicalSectionInds]
#        listOfObliqueSections = [cell.apic[x] for x in listOfRandObliqueSectionInds]
        #listOfApicalSections = []
        
#        listOfSections = listOfApicalSections + listOfBasalSections
        
        listOfRandBasalLocationsInSection = np.random.rand(len(listOfRandBasalSectionInds))
        listOfRandApicalLocationsInSection = np.random.rand(len(listOfRandApicalSectionInds))
#        listOfRandObliqueLocationsInSection = np.random.rand(len(listOfRandObliqueSectionInds))
#        listOfSegLocs = list(listOfRandApicalLocationsInSection) + list(listOfRandBasalLocationsInSection)
        #listOfSegLocs = list(listOfRandBasalLocationsInSection)
        
        listOfEvents = []
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 2))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + 100*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 2))
        
        for k, section in enumerate(listOfApicalSections):
            eventTime = silentTimeAtStart + delayTime + treeT*np.random.normal(0,1) #gauss(0.5,0.2)
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandApicalLocationsInSection[k]), eventTime, 2))
        
        for k, section in enumerate(listOfBasalSections):
            eventTime = silentTimeAtStart + delayTime + ApiBasInt + treeT*np.random.normal(0,1) #simulationTime/2*np.random.rand(1)[0]
            listOfEvents.append(Add_NMDA_SingleSynapticEventToSegment(section(listOfRandBasalLocationsInSection[k]), eventTime, 2))
        
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
                tempSpikes += 1
                k = k+origNumSamplesPerMS*3
            else:
                k = k+5 # was 1 before
        
        spikes.append(origSpikes)
        if tempSpikes>1: 
            freq[experiment] = tempSpikes/(origSpikes[-1]-origSpikes[-tempSpikes])
            for spk in range(tempSpikes-1):
                isis.append(origSpikes[-1-spk]-origSpikes[-2-spk])
    #    plt.figure()
    #    plt.plot(recordingTime, somaVoltage)
    #    plt.xlabel('Time [ms]'); plt.ylabel('Voltage [mV]')
    #    plt.axis(xmin=0, xmax=stopTime, ymin=min(somaVoltage)-5, ymax=max(somaVoltage)+5)
        
        #listOfEvents = []
        if (experiment+1)%10==0 or (time.time()-startTime)/60>5 or numExp<5: 
            print "Dt %s treeTime %s exp. # %s took %.3f minutes" % (ApiBasInt,treeT,experiment+1, (time.time()-startTime)/60)
        
    print "Mean no. of spikes: %s" % (float(numSpikes)/numExp)
    return float(numSpikes)/numExp,np.mean(freq),isis#, listOfSomaTraces, recordingTime

#%% run simulation on some parameter pair, plot the space
L5PC = h.L5PCtemplate(morphologyFilename)
saveDir = 'C:/Users/Leleo/Documents/burst/spksfreq'#'/ems/elsc-labs/segev-i/eilam.goldenberg/Documents/coincidence/tot_api/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

try:
    randomSeed = int(sys.argv[1])
    print 'random seed selected by user - %d' %(randomSeed)
except:
    randomSeed = np.random.randint(100000)
    print 'randomly chose seed - %d' %(randomSeed)

np.random.seed(randomSeed)

#dts = np.linspace(-50,-25,num=6),np.linspace(-20,20,num=21),np.linspace(25,100,num=16)
ApicalBasalInterval = 0 #np.linspace(-30,40,num=71) 
numBasal = np.linspace(0,200,num=5)#11) #41)
numApical = np.linspace(0,300,num=7)#20,40,num=21) #200-numBasal 
#numOblique = 40-numApical
totalSyn = [20,50,100,200]#np.linspace(0,200,num=41)
partApical = 10 #0.1*np.logspace(0,10,num=31,base=2)
medSegment = 60 
treeTime = 2 #np.logspace(0,4,num=17,base=2)
numExperiments=20

spks = [[0 for i in range(len(numApical))] for j in range(len(numBasal))] 
frqs = [[0 for i in range(len(numApical))] for j in range(len(numBasal))] 
isis = [[0 for i in range(len(numApical))] for j in range(len(numBasal))]
#trc = [[[] for i in range(len(numApical))] for j in range(len(numBasal))]

i = 0
j = 0

start = time.time()

#for ApiBasInd in ApicalBasalInterval:
#    print "Running for interval: %s [ms]" % (int(ApiBasInd))
#for totalS in totalSyn:
for numB in numBasal:
#    for treeT in treeTime:
    for numA in numApical:#np.linspace(0,totalS,num=21):
#        numA = numApical
        print "Running for basal # %s apical # %s" % (int(numB),int(numA))
#        numA = int(totalS*0.4)
        spks[j][i],frqs[j][i],isis[j][i] = runSim(L5PC,ApicalBasalInterval,treeTime,numB,numA,partApical,medSegment,numExperiments)
        j = j+1
    j = 0
    i = i+1
    
pickle.dump(spks,open(saveDir+'bas_api_treet2_dt0_spks'+str(randomSeed)+".npy","wb"),protocol=2)
pickle.dump(isis,open(saveDir+'bas_api_treet2_dt0_isis'+str(randomSeed)+".npy","wb"),protocol=2)

print "Total running time was: ", (time.time()-start)/3600, "hours"
#%% scatter
y = [[0 for i in range(len(spks[0]))] for j in range(len(spks))]
z = [[0 for i in range(len(spks[0]))] for j in range(len(spks))]
for i in range(len(spks[0])): 
    for j in range(len(spks)):
        if not isis[j][i]:# == []:
            y[j][i] = 0
            z[j][i] = 0
#        elif isis[j][i]>0:
#            y[j][i] = isis[j][i]
#            z[j][i] = isis[j][i]            
        else:            
            y[j][i] = min(isis[j][i])
            z[j][i] = max(isis[j][i])
        

#%% plot freq in heatmap #3D
ind0+=1
fig = plt.figure('w by dist & thr #%s' %(ind0))#frqs bas-#api(max50) dt=0 SD=100 partA=5 medS=60 2D #%s' % (ind0))# medS=60')
#                 'Spikes-dt-clustSize len=10 SD=20 tot=100 medS=60 #%s' % (ind0)) 
ax = fig.gca()#projection='3d')
plt.title('FR as function of basal/apical synapses') 
    #$\Delta$t and synapses per cluster')# % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
plt.xlabel('# of apical synapses');plt.ylabel('# of basal synapses')#$\Delta$t'); plt.ylabel('synapses/cluster')#$\sigma$')
#plt.ylabel('Change in weight (norm.) by tuft loc and $\theta$_p'); plt.xlabel('location on apical tuft')#basal synapses')
plt.axis(xmin=min(numBasal), xmax=max(numBasal), ymin=min(numApical), ymax=max(numApical))#ymin=1/max(partApical), ymax=1/min(partApical))
#ax.set_xlabel([0,300])#'Intraburst firing rate')#'mean # of spikes')#
apbas4plt = [[ind for ind in numBasal] for j in range(len(numApical[:-10]))]#ApicalBasalInterval] for j in range(len(partApical))]#
numap4plt = [[j for ind in numBasal] for j in range(len(numApical[:-10]))]#range(len(ApicalBasalInterval))] for j in partApical]
#plt.imshow(log2w1,cmap=cm.jet,extent=[0,40,0.1,8.1],origin='lower',aspect='auto')
surf = ax.plot_surface(apbas4plt,numap4plt,np.asarray(frqsmean),cmap=cm.jet)#,vmin=0,vmax=4)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.colorbar()
plt.show()

#%% plot lines in color
ind+=1
plt.figure('Spikes-dt std %s ms tot=30 partApi %s medSeg=62 # %s' 
          % (treeT,partApi,ind) )
#plt.grid()
#plt.figure('Spikes by inh proximal-distal from exc 120excdt0sd0meds62 # %s' % (ind)) 
plt.title('Spikes by dt & sd')#inh syn #')# timing/proximal-distal from exc')# on section # and $\Delta$t') # % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
plt.xlabel('$\Delta$t [ms]')#to inhibition [ms]'); 
#plt.xlabel('inhibitory synapses distance from soma [microm]'); plt.ylabel('mean # of spikes')
plt.axis(xmin=min(ApicalBasalInterval), xmax=max(ApicalBasalInterval))#, ymin=min(medSegment), ymax=max(medSegment))
#plt.title('Spikes by apical tree part')#synapse # per cluster')#$\Delta$t - No VGCC') # % (numExperiments, partOfApical, numApical, numBasal, treeTime, ApicalBasalInterval) )
#plt.xlabel('% of apical tree'); plt.ylabel('mean # of spikes')
#lines = [0 for x in range(10)]#len(clusterSize))]#totalSyn))]
colors = cm.get_cmap('coolwarm')#('YlGnBu')#jet')#('cool')#
j = 0
for i in [10,12,13,14]:#np.linspace(0,10,num=6):#range(1):#len(lines)):#[8,9,16,18,20]:#3,5,7,9,11,13]:#np.linspace(0,20,num=11):#
#    lines[j]=plt.plot(np.linspace(0,100,num=41),spksnew0[int(i)],'.-',label='%.2f' % totalSyn[i],color=colors(float(i)/len(lines)))
    lines[j]=plt.plot(ApicalBasalInterval,spks[int(i)],'.-',label='%s' % int(treeTime[int(i)]),color=colors(float(3-j)/3))#(float(i)-10)/4))#len(lines)))
    j += 1
plt.legend()#bbox_to_anchor=(0.9,0.6)) #lines,['2.26','3.2','4.53','6.4','9.05'])

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
    
    