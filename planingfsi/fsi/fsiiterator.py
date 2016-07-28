import os
from fnmatch import fnmatch

import numpy as np

import planingfsi.config as config
import planingfsi.krampy as kp

from fsifigure import FSIFigure

class Simulation:

    def __init__(self, solid, fluid, dictName='configDict'):
        self.solid = solid
        self.fluid = fluid

    def setFluidPressureFunc(self, func):
        self.fluidPressureFunc = func

    def setSolidPositionFunc(self, func):
        self.solidPositionFunc = func

    def updateFluidResponse(self):
        self.fluid.calculateResponse()
        self.solid.updateFluidForces()

    def updateSolidResponse(self):
        self.solid.calculateResponse()

    def createDirs(self):
        config.itDir  = os.path.join(config.caseDir, '{0}'.format(config.it))
        config.figDir = os.path.join(config.caseDir, config.figDirName)

        if self.checkOutputInterval() and not config.resultsFromFile:
            kp.createIfNotExist(config.caseDir)
            kp.createIfNotExist(config.itDir)

        if config.plotSave:
            kp.createIfNotExist(config.figDir)
            if config.it == 0:
                for f in os.listdir(config.figDir):
                    os.remove(os.path.join(config.figDir, f))

    def run(self):
        config.it = 0
        self.solid.generateMesh()

        if config.resultsFromFile:
            self.createDirs()
            self.applyRamp()
            self.itDirs = kp.sortDirByNum([d for d in os.listdir(config.caseDir) if fnmatch(d, '[0-9]*')])[1]

        if config.plot:
            self.figure = FSIFigure(self.solid, self.fluid)
        
        # Initialize body at specified trim and draft 
        self.solid.initializeRigidBodies()
        self.updateFluidResponse()
        
        # Iterate between solid and fluid solvers until equilibrium
        while config.it <= config.rampIt \
            or (self.getResidual() >= config.maxRes \
                and config.it <= config.maxIt):
            
            # Calculate response
            if config.hasFreeStructure:
                self.applyRamp()
                self.updateSolidResponse()
                self.updateFluidResponse()
                self.solid.getResidual()
            else:
                self.solid.res = 0.0
          
            # Write, print, and plot results  
            self.createDirs()
            self.writeResults()
            self.printStatus()
            self.updateFigure()

            # Increment iteration count
            self.increment()

        if config.plotSave and not config.figFormat == 'png':
            config.figFormat = 'png'
            self.figure.save()

        if config.writeTimeHistories:
            self.figure.writeTimeHistories()
        
        print "Execution complete"
       
        if config.plotShow:
            self.figure.show()

    def increment(self):      
        if config.resultsFromFile:
            oldInd = np.nonzero(config.it == self.itDirs)[0][0]
            if not oldInd == len(self.itDirs)-1:
                config.it = int(self.itDirs[oldInd+1])
            else:
                config.it = config.maxIt + 1 
            self.createDirs()
        else:
            config.it += 1

    def updateFigure(self):
        if config.plot:
            self.figure.update()

    def getBodyRes(self, x):
        self.solid.getPtDispRB(x[0], x[1])
        self.solid.updateNodalPositions()
        self.updateFluidResponse()

        # Write, print, and plot results  
        self.createDirs()
        self.writeResults()
        self.printStatus()
        self.updateFigure()

        # Update iteration number depending on whether loading existing or simply incrementing by 1
        if config.resultsFromFile:
            if it < len(self.itDirs) - 1:
                config.it = int(self.itDirs[it+1])
            else:
                config.it = config.maxIt
            it += 1
        else:
            config.it += 1
      
        config.resL = self.solid.rigidBody[0].getResL()
        config.resM = self.solid.rigidBody[0].getResM()
       
        print 'Rigid Body Residuals:'
        print '  Lift:   {0:0.4e}'.format(config.resL)
        print '  Moment: {0:0.4e}\n'.format(config.resM)
               
        return np.array([config.resL, config.resM])
   
    def applyRamp(self):
        if config.resultsFromFile:
            self.loadResults()
        else:
            if config.rampIt == 0:
                ramp = 1.0
            else:
                ramp = np.min((config.it / float(config.rampIt), 1.0))

            config.ramp     = ramp
            config.relaxFEM = (1 - ramp) * config.relaxI + ramp * config.relaxF

    def getResidual(self):
        if config.resultsFromFile:
            return 1.0
            return kp.Dictionary(os.path.join(config.itDir,'overallQuantities.txt')).readOrDefault('Residual',0.0)
        else:
            return self.solid.res

    def printStatus(self):
        print 'Residual after iteration {1:>4d}: {0:5.3e}'.format(self.getResidual(), config.it)

    def checkOutputInterval(self):
        return config.it >= config.maxIt or \
               self.getResidual() < config.maxRes or \
               np.mod(config.it, config.writeInterval) == 0
    
    def writeResults(self):
        if self.checkOutputInterval() and not config.resultsFromFile:
            kp.writeasdict(os.path.join(config.itDir, 'overallQuantities.txt'), ['Ramp', config.ramp], ['Residual', self.solid.res])

            self.fluid.writeResults()
            self.solid.writeResults()

    def loadResults(self):
        K = kp.Dictionary(os.path.join(config.itDir, 'overallQuantities.txt'))
        config.ramp = K.readOrDefault('Ramp', 0.0)
        self.solid.res = K.readOrDefault('Residual', 0.0)
