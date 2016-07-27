import numpy as np
import krampy as kp
import matplotlib.pyplot as plt
import os, time

class Case:

  def __init__(self):
    self.readCaseInfo()
    kp.checkDir('dataPost')
    kp.checkDir('figsPost')

    self.Data_f  = []
    self.Data_p  = []
    self.Data_t  = []
    self.Data_fs = []
  
  def readCaseInfo(self):
    f = open('caseInfo')
    line = f.readline()
    while line:
      word = line.split()
      exec('self.' + word[0] + ' = ' + word[1])
      line = f.readline()
    f.close()
    self.Fr  = self.U / np.sqrt(2 * self.g * self.Li)
    self.q   = 0.5 * self.rho * self.U**2
 
  def loadForceData(self):
    forceDir = 'plateForce'
    timeDir = kp.listdirNoHidden(forceDir)
    Data = []
    for i in range(0,len(timeDir)):
      f = open(os.path.join(forceDir, timeDir[i], 'forces.dat'))
      tmpFName = 'tmpForceRead.dat'
      fTmp = open(tmpFName,'w')
      dataRaw = f.read()
      dataRaw = dataRaw.replace('(',' ')
      dataRaw = dataRaw.replace(')',' ')
      f.close()
      fTmp.write(dataRaw)
      fTmp.close()

      DataNew = np.loadtxt(tmpFName)
      DataNew = DataNew[:,0:13]
      os.remove(tmpFName)
      if Data == []:
        Data = np.zeros((1,DataNew.shape[1]))
      Data = np.append(Data, DataNew, axis=0)
    I = np.argsort(Data[:,0])
    self.Data_f = Data[I,:]
 
  def loadSurfaceData(self):
    # Import data from files
    surfDir   = 'surfaceSampling'
    surfName  = 'plateSurfaces'

    timeDir   =  kp.listdirNoHidden(surfDir)
    timeFloat = np.zeros(len(timeDir))
    for i in range(len(timeFloat)):
      timeFloat[i] = float(timeDir[i])
    I = np.argsort(timeFloat)
    self.time = timeFloat[I]    
    self.tMax = np.max(self.time)

    if os.path.exists(os.path.join(surfDir, timeDir[I[0]], 'pMean_{0}.raw'.format(surfName))):
      pressureVar = 'pMean'
    else:
      pressureVar = 'p'
    
    print 'Using {0} for pressure variable'.format(pressureVar)

    shearVar = 'wallGradU'
  
    for i in range(len(I)):
      timeStr = timeDir[I[i]]
      self.Data_p.append(np.loadtxt(os.path.join(surfDir, timeStr, '{0}_{1}.raw'.format(pressureVar, surfName))))
      self.Data_t.append(np.loadtxt(os.path.join(surfDir, timeStr, '{0}_{1}.raw'.format(shearVar, surfName))))

  def loadFreeSurfaceData(self):
    # Import data from files
    baseName = 'surfaceSampling'
    timeDir   =  kp.listdirNoHidden(baseName)
    timeFloat = np.zeros(len(timeDir))
    for i in range(len(timeFloat)):
      timeFloat[i] = float(timeDir[i])
    I = np.argmax(timeFloat)
    timeStr = timeDir[I]
    self.Data_fs = np.loadtxt(os.path.join(baseName, timeStr, 'p_freeSurface.raw'))
 
  def calculateForces(self):
    if self.Data_f == []:
      self.loadForceData()

    # Extract desired values from imported data
    self.ft = self.Data_f[:,0]
    self.fD = self.Data_f[:,1] + self.Data_f[:,4]
    self.fL = self.Data_f[:,2] + self.Data_f[:,5]
    self.fM = self.Data_f[:,9] + self.Data_f[:,12]
 
  def calculateWettedLength(self):
    if self.Data_p == []:
      self.loadSurfaceData()

    # Extract desired values from imported data
    x = self.Li - self.Data_p[-1][:,0] / kp.cosd(self.AOA)
    p = self.Data_p[-1][:,3]
    t = self.nu * self.rho * (kp.cosd(self.AOA) * self.Data_t[-1][:,3] - kp.sind(self.AOA) * self.Data_t[-1][:,4])

    # Sort data
    I = np.argsort(-x)
    self.x = x[I]
    self.p = p[I]
    self.t = t[I]

    # Post-process pressure profile
    self.Lw   = np.zeros(len(self.time))
    self.pBar = np.zeros(len(self.time))
    self.pTE  = np.zeros(len(self.time))
    self.xBar = np.zeros(len(self.time))
    for i in range(len(self.time)):
      # Extract desired values from imported data
      x = self.Li - self.Data_p[i][:,0] / kp.cosd(self.AOA)
      p = self.Data_p[i][:,3]
      t = self.nu * self.rho * (kp.cosd(self.AOA) * self.Data_t[i][:,3] - kp.sind(self.AOA) * self.Data_t[i][:,4])

      # Sort data
      I = np.argsort(-x)
      x = x[I]
      p = p[I]
      t = t[I]

      # Interpolate shear stress to find stagnation point
      self.Lw[i]   = np.interp(0, t, x)
      P            = integrate(p, x)
      self.pBar[i] = P / self.Lw[i]
      self.pTE[i]  = p[-1]
      self.xBar[i] = integrate(p * x, x) / P

  def calculateFreeSurfaceElevation(self):
    if self.Data_fs == []:
      self.loadFreeSurfaceData()

    xFS = self.Data_fs[:,0]
    yFS = self.Data_fs[:,1]

    I = np.argsort(xFS)
    xFS = xFS[I]
    yFS = yFS[I]
   
    I = np.nonzero(xFS < 0)
    xFSL, yFSL = sortPts(xFS[I], yFS[I])
   
    I = np.nonzero(xFS > 0)
    xFSR = xFS[I]
    yFSR = yFS[I] 

    self.xFS = np.hstack([xFSL,xFSR])
    self.yFS = np.hstack([yFSL,yFSR])

  def calculateTimeAverages(self):
    # Calculate max and min for plotting
    xfD    = 1 / (self.q * self.Li)
    xfL    = 1 / (self.q * self.Li)
    xfM    = 1 / (self.q * self.Li**2)
    xLw    = 1 / (4 * self.Li)
    xxBar  = 1 / (4 * self.Li)
    xpBar  = 1 / self.q 
    xpTE   = xpBar

    plotT = ['ft', 'ft',  'ft',  'time', 'time', 'time', 'time']
    plotF = ['fD', 'fL',  'fM',  'Lw',   'xBar', 'pBar', 'pTE']
    plotS = ['k-', 'b--', 'r-.', 'g-',   'c-',   'm--',  'y-']

    yMax = 0
    yMin = 0
    fig = plt.figure()
    for i in range(len(plotF)):
      # Calculate scaled value
      exec('C{0} = self.{1} * x{2}'.format(plotF[i], plotF[i], plotF[i]))

      # Plot Value
      exec('plt.plot(self.{0}, C{1}, \'{2}\')'.format(plotT[i], plotF[i], plotS[i]))
      
      # Calculate maximum and minimum
      exec('yMin = np.min([yMin,np.min(C{0}[-1])])'.format(plotF[i]))
      exec('yMax = np.max([yMax,np.max(C{0}[-1])])'.format(plotF[i]))

    # Plot raw time histories
    yMin -= 0.25 * np.abs(yMax)
    yMax += 0.25 * np.abs(yMax)
    plt.xlabel(r'$t\,\mathrm{[s]}$', size=16)
    plt.ylim([yMin, yMax])
    plt.legend([r'$C_D$', r'$C_L$', r'$C_M$', r'$0.25L_w/L_i$', r'$0.25\bar{x}/L_i$', r'$C_\bar{p}$'],loc='upper left')
 
    # Choose point to begin time-averaging by clicking the plot
    print 'Please select starting point for time integration:'
    pt = plt.ginput(1)
    t0 = map(lambda x: x[0], pt)
    print 'Selected t =', t0[0]
     
    # Average data over selected time range
    for i in range(len(plotF)):
      exec('self.{0}_avg = average(self.{1}[self.{2} > t0], self.{3}[self.{4} > t0])'.format(\
           plotF[i], plotF[i], plotT[i], plotT[i], plotT[i]))
      exec('self.{0}_std = np.std(self.{1}[self.{2} > t0])'.format(\
           plotF[i], plotF[i], plotT[i]))
      exec('C{0}_avg = self.{1}_avg * x{2}'.format(\
           plotF[i], plotF[i], plotF[i]))
      exec('plt.plot(self.{0}, C{1}_avg * np.ones_like(self.{2}), \'{3}\')'.format(\
           plotT[i], plotF[i], plotT[i], plotS[i]))
      exec('plt.plot(t0, C{0}_avg, \'{1}o\')'.format(\
           plotF[i], plotS[i][0]))

    plt.ylim([yMin, yMax])
    plt.savefig('./figsPost/timeHistories.eps', format='eps')
    plt.draw()
    time.sleep(2)
    plt.close('all') 

  def writeFreeSurfaceElevation(self):
    f = open('./dataPost/freeSurface.dat','w')
    f.write('# xFS [m], yFS [m] \n')
    for i in range(self.xFS.shape[0]):
      f.write('{0:10.8e} {1:10.8e}\n'.format(self.xFS[i], self.yFS[i]))
    f.close()


  def writeTimeAverages(self):
    # Write results to .dat file
    f = open('./dataPost/timeAverages.dat','w')
    rows = [['# U[m/s]', 'tMax[s]', 'D[N]',       'L[N]',      'M[N-m]',    'Lw[m]',     'pBar[Pa]',    'xBar[m]',     'pTE[Pa]'], \
            [self.U,      self.tMax, self.fD_avg, self.fL_avg, self.fM_avg, self.Lw_avg, self.pBar_avg, self.xBar_avg, self.pTE_avg], \
            [0,           0,         self.fD_std, self.fL_std, self.fM_std, self.Lw_std, self.pBar_std, self.xBar_std, self.pTE_std]]
    rowFormats = ['s','10.4f','10.4f']
    for row, rowFormat in zip(rows, rowFormats):
      for val in row:
        f.write('{0:{1}} '.format(val, rowFormat))
      f.write('\n')
    f.close()


  def writeTimeHistories(self):
    # Write results to .dat file
    f = open('./dataPost/forceTimeHistories.dat','w')
    f.write('# t [s] D [N] L [N] M [N-m]\n')
    for t, D, L, M in zip(self.ft, self.fD, self.fL, self.fM):
      for i in [t,D,L,M]:
        f.write('{0:8.6e} '.format(i))
      f.write('\n')
    f.close()
    # Write results to .dat file
    f = open('./dataPost/pressureTimeHistories.dat','w')
    f.write('# t [s] Lw [m] pBar [Pa] xBar [m] pTE [Pa]\n')
    for t, Lw, pBar, xBar, pTE in zip(self.time, self.Lw, self.pBar, self.xBar, self.pTE):
      for i in [t,Lw,pBar,xBar,pTE]:
        f.write('{0:8.6e} '.format(i))
      f.write('\n')
    f.close()


  def writePressureAndShear(self):
    # Write rescaled pressure and shear stress data to file
    f = open('./dataPost/pressureAndShear.dat','w')
    f.write('# x [m], p [Pa], tau [Pa] \n')
    for i in range(0,self.x.shape[0]):
      f.write(str(self.x[i]) + ' ' + str(self.p[i]) + ' ' + str(self.t[i]) + ' \n')
    f.close()


def sortPts(x,y):
  x1 = x[1:]
  y1 = y[1:]
  x2 = np.array([x[0]])
  y2 = np.array([y[0]])
  it = 0
  while len(x1) > 0 and it < len(x)+10:
    d = (x1 - x2[-1])**2 + (y1 - y2[-1])**2
    I = np.argmin(d)

    x2 = np.hstack([x2,x1[I]])
    y2 = np.hstack([y2,y1[I]])

    x1 = np.hstack([x1[0:I],x1[I+1:]])
    y1 = np.hstack([y1[0:I],y1[I+1:]])

    it += 1

  return x2, y2
def average(f, x):
  F = integrate(f,x)
  return F / (np.max(x) - np.min(x))
def integrate(f, x):
  I = np.argsort(x)
  x = x[I]
  f = f[I]
  return 0.5 * np.sum((x[1:] - x[0:-1]) * (f[1:] + f[0:-1]))
