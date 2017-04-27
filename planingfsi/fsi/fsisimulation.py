import os

import numpy as np

import planingfsi.config as config
import planingfsi.krampy as kp

from planingfsi.potentialflow.solver import PotentialPlaningSolver
from planingfsi.fe.structure import FEStructure

from fsifigure import FSIFigure


class Simulation(object):
    """Simulation object to manage the FSI problem. Handles the iteration
    between the fluid and solid solvers.

    Attributes
    ----------
    solid : FEStructure
        Solid solver.

    fluid : PotentialPlaningSolver
        Fluid solver.
    """

    def __init__(self):

        # Create solid and fluid solver objects
        self.solid = FEStructure()
        self.fluid = PotentialPlaningSolver()

#     def setFluidPressureFunc(self, func):
#         self.fluidPressureFunc = func
#
#     def setSolidPositionFunc(self, func):
#         self.solidPositionFunc = func

    def updateFluidResponse(self):
        self.fluid.calculate_response()
        self.solid.update_fluid_forces()

    def updateSolidResponse(self):
        self.solid.calculate_response()

    def createDirs(self):
        config.it_dir = os.path.join(config.case_dir, '{0}'.format(config.it))
        config.fig_dir_name = os.path.join(
            config.case_dir, config.fig_dir_name)

        if self.checkOutputInterval() and not config.resultsFromFile:
            kp.createIfNotExist(config.case_dir)
            kp.createIfNotExist(config.it_dir)

        if config.plotSave:
            kp.createIfNotExist(config.fig_dir_name)
            if config.it == 0:
                for f in os.listdir(config.fig_dir_name):
                    os.remove(os.path.join(config.fig_dir_name, f))

    def run(self):
        """Run the fluid-structure interaction simulation by iterating
        between the fluid and solid solvers.
        """
        config.it = 0
        self.solid.load_mesh()

        if config.resultsFromFile:
            self.createDirs()
            self.applyRamp()
            self.itDirs = kp.sortDirByNum(kp.find_files('[0-9]*'))[1]

        if config.plot:
            self.figure = FSIFigure(self.solid, self.fluid)

        # Initialize body at specified trim and draft
        self.solid.initialize_rigid_bodies()
        self.updateFluidResponse()

        # Iterate between solid and fluid solvers until equilibrium
        while config.it <= config.rampIt \
            or (self.get_residual() >= config.maxRes and
                config.it <= config.maxIt):

            # Calculate response
            if config.has_free_structure:
                self.applyRamp()
                self.updateSolidResponse()
                self.updateFluidResponse()
                self.solid.get_residual()
            else:
                self.solid.res = 0.0

            # Write, print, and plot results
            self.createDirs()
            self.write_results()
            self.printStatus()
            self.updateFigure()

            # Increment iteration count
            self.increment()

        if config.plotSave and not config.fig_format == 'png':
            config.fig_format = 'png'
            self.figure.save()

        if config.writeTimeHistories:
            self.figure.writeTimeHistories()

        print "Execution complete"

        if config.plotShow:
            self.figure.show()

    def increment(self):
        if config.resultsFromFile:
            oldInd = np.nonzero(config.it == self.itDirs)[0][0]
            if not oldInd == len(self.itDirs) - 1:
                config.it = int(self.itDirs[oldInd + 1])
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
        self.write_results()
        self.printStatus()
        self.updateFigure()

        # Update iteration number depending on whether loading existing or
        # simply incrementing by 1
        if config.resultsFromFile:
            if it < len(self.itDirs) - 1:
                config.it = int(self.itDirs[it + 1])
            else:
                config.it = config.maxIt
            it += 1
        else:
            config.it += 1

        config.resL = self.solid.rigid_body[0].get_res_l()
        config.resM = self.solid.rigid_body[0].get_res_moment()

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

            config.ramp = ramp
            config.relaxFEM = (1 - ramp) * config.relaxI + ramp * config.relaxF

    def get_residual(self):
        if config.resultsFromFile:
            return 1.0
            return kp.Dictionary(os.path.join(config.it_dir, 'overallQuantities.txt')).read('Residual', 0.0)
        else:
            return self.solid.res

    def printStatus(self):
        print 'Residual after iteration {1:>4d}: {0:5.3e}'.format(self.get_residual(), config.it)

    def checkOutputInterval(self):
        return config.it >= config.maxIt or \
            self.get_residual() < config.maxRes or \
            np.mod(config.it, config.writeInterval) == 0

    def write_results(self):
        if self.checkOutputInterval() and not config.resultsFromFile:
            kp.writeasdict(os.path.join(config.it_dir, 'overallQuantities.txt'), [
                           'Ramp', config.ramp], ['Residual', self.solid.res])

            self.fluid.write_results()
            self.solid.write_results()

    def loadResults(self):
        K = kp.Dictionary(os.path.join(config.it_dir, 'overallQuantities.txt'))
        config.ramp = K.read('Ramp', 0.0)
        self.solid.res = K.read('Residual', 0.0)
