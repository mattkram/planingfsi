import numpy as np
import matplotlib.pyplot as plt


class Mesh:

  def __init__(self, fName):
    self.point = []
    self.line   = []
    self.block  = []
    self.boundary = []
    self.makePoint(0, 'dir', [0, 0])
    self.sf = 1.0
    self.conversionFactor = 1.0
    self.numCellsR = []
    self.numCellsC = []
    self.f = open(fName, 'w')

  def readCaseInfo(self):
    # Custom Parameters
    # Load in AOA from caseInfo file
    f = open('caseInfo')
    line = f.readline()
    while line:
      word = line.split()
      exec('self.' + word[0] + ' = ' + word[1])
      line = f.readline()
    f.close()

  def makePoint(self, ID, method, position):
    P = Point(ID, len(self.point))

    if   method == 'dir':
      P.pos  = np.array(position)
    elif method == 'rel':
      basePt = position[0]
      ang    = position[1]
      R      = position[2]
      P.pos  = self.getPointPos(basePt) + R * ang2vec(ang)
    elif method == 'con':
      basePt    = position[0]
      dimension = position[1]
      value     = position[2]
      if len(position) == 3:
        if dimension == 'x':
          ang = 0
        else:
          ang = 90
      else:
        ang = position[3]

      if   dimension == 'x':
        P.pos[0] = value
        P.pos[1] = self.getPointPos(basePt)[1] + (value - self.getPointPos(basePt)[0])*tand(ang)
      elif dimension == 'y':
        P.pos[1] = value
        P.pos[0] = self.getPointPos(basePt)[0] + (value - self.getPointPos(basePt)[1])/tand(ang)
      else:
        print 'Incorrect dimension specification'
    else:
      print 'Incorrect position specification method for point', ID

    print 'Point', P.ind, ': ID =', P.ID, 'Pos =', P.pos
    self.point.append(P)

  def makeLine(self, ID, pt1, pt2, grading):
    L = Line(ID, len(self.line), pt1, pt2, grading)
    for i in range(len(L.ptID)):
      L.pts.append(self.point[self.getIndex(self.point, L.ptID[i])])
      L.ptInd.append(L.pts[i].ind)

    print 'Line', L.ind, ': ID =', L.ID, 'Point IDs', L.ptID, 'Point Inds', L.ptInd
    self.line.append(L)

  def makeBlock(self, name, lineID, baseCrnr, blockPos):
    B = Block(name, len(self.block), lineID)
    if baseCrnr == 1 or baseCrnr == 3:
      B.nPts = [int(self.numCellsC[blockPos[1] - 1]*self.sf), \
                int(self.numCellsR[blockPos[0] - 1]*self.sf)]
    else:  
      B.nPts = [int(self.numCellsR[blockPos[0] - 1]*self.sf), \
                int(self.numCellsC[blockPos[1] - 1]*self.sf)]
   
    # Find point objects and indices corresponding to the line endpoints
    pts = []
    ptID = []
    for i in range(0,len(B.lineID)):
      # Get point objects associated with line i
      ptTmp = self.getLinePts(B.lineID[i])
      ptID.append(ptTmp[0].ID)
      ptID.append(ptTmp[1].ID)
      B.lineInd.append(self.getIndex(self.line, B.lineID[i]))
      B.lines.append(self.line[B.lineInd[i]])
    
    # Place first two points in correct order
    orderedPtID = []
    if ptID[0] == ptID[-1] or ptID[0] == ptID[-2]:
      orderedPtID.append(ptID[0])
      orderedPtID.append(ptID[1])
    else:
      orderedPtID.append(ptID[1])
      orderedPtID.append(ptID[0])

    # Order remaining points
    for i in range(2,len(ptID)):
      match = False
      for j in range(0,i):
        if ptID[j] == ptID[i]:
          match = True
      if not match:
        orderedPtID.append(ptID[i])

    # Order points again based on base corner specification 
    for i in orderedPtID[baseCrnr-1:]:
      B.ptID.append(i)
    for i in orderedPtID[0:baseCrnr-1]:
      B.ptID.append(i)
    
    # Assign lists of point objects and indices for each point ID
    for i in range(len(B.ptID)):
      index = self.getIndex(self.point, B.ptID[i])
      B.ptInd.append(index)
      B.pts.append(self.point[index])

 
    # Find proper order for grading of the lines and assign to grading
    if baseCrnr == 1:
      lineInd = [0,2,2,0,3,1,1,3]
    elif baseCrnr == 2:
      lineInd = [1,3,3,1,0,2,2,0]
    elif baseCrnr == 3:
      lineInd = [2,0,0,2,1,3,3,1]
    else:
      lineInd = [3,1,1,3,2,0,0,2]

    for i in lineInd:
      B.grading.append(self.line[self.getIndex(self.line, B.lineID[i])].grading)
    
    # Add additional 4 points to make 3d
    nPts = len(B.ptID)
    for i in range(0,nPts):
      B.ptID.append(B.ptID[i]+len(self.point))
      B.ptInd.append(B.ptInd[i]+len(self.point))

    print 'Block', B.ind, ':', 'Name =', B.name, 'Point IDs', B.ptID, 'Point Inds', B.ptInd
    self.block.append(B)

  def makeBoundary(self, name, patchType, lineID):
    B = Boundary(name, len(self.boundary), patchType, lineID)
    if not patchType == 'empty':
      for i in range(len(B.lineID)):
        lineInd = self.getIndex(self.line, np.abs(B.lineID[i]))
        B.lineInd.append(lineInd)
        
        pts = []
        if B.lineID[i] < 0:
          pts.append(self.line[lineInd].ptInd[0])
          pts.append(self.line[lineInd].ptInd[1])
        else:
          pts.append(self.line[lineInd].ptInd[1])
          pts.append(self.line[lineInd].ptInd[0])
        pts.append(pts[1] + len(self.point))
        pts.append(pts[0] + len(self.point))
        
        B.facePts.append(pts)
    else:
      for i in range(len(self.block)):
        pts = self.block[i].ptInd[4:]
        B.facePts.append(pts)
        pts = self.block[i].ptInd[0:4]
        pts.reverse()
        B.facePts.append(pts)

    print 'Boundary', B.ind, ':', 'Name =', B.name, 'Type =', B.patchType, 'Line ID =', B.lineID, 'Line Ind =', B.lineInd
    self.boundary.append(B)

  def rotatePoints(self, IDlist, basePtID, angle):
    print IDlist
    basePt = self.point[self.getIndex(self.point, basePtID)].pos
    for i in range(len(IDlist)):
      P = self.point[self.getIndex(self.point, IDlist[i])]
      xBar = P.pos[0] - basePt[0]
      yBar = P.pos[1] - basePt[1]
      
      P.pos[0] = xBar*cosd(angle) - yBar*sind(angle) + basePt[0]
      P.pos[1] = xBar*sind(angle) + yBar*cosd(angle) + basePt[1]

  def getPointPos(self, ID):
    return self.point[self.getIndex(self.point, ID)].pos

  def setPointPos(self, ID, pos):
    self.point[self.getIndex(self.point, ID)].pos = pos

  def getLinePts(self, ID):
    return self.line[self.getIndex(self.line, ID)].pts

  def getIndex(self, array, ID):
    for i in range(len(array)):
      if array[i].ID == ID:
        return array[i].ind
    return -1

#  def makeEdge(self):

  def plotMesh(self, **kwargs):
    plt.figure()
    
    # Plot lines
    for i in range(len(self.line)):
      self.line[i].plot()

    # Plot blocks
    for i in range(len(self.block)):
      self.block[i].plot()
    
    # Plot vertices
    for i in range(len(self.point)):
      self.point[i].plot()

    plt.axis('equal')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Process optional arguments and save or show figure
    showTF = kwargs.get('show',False)
    saveTF = kwargs.get('save',False)
    savedFileName = kwargs.get('fileName','meshLayout')

    if saveTF:
      plt.savefig(savedFileName + '.eps', format='eps')
    if showTF:
      plt.show()

  def writeBlockMeshDict(self):
    self.writeHeader()
    self.writeConversionFactor()
    self.writePoints()
    self.writeBlocks()
    self.writeEdges()
    self.writeBoundaries()
    self.writeClosing()
    self.f.close()

  def writeHeader(self):
    self.f.write('/*--------------------------------*- C++ -*----------------------------------*\\\n' \
                 '| =========                 |                                                 |\n' \
                 '| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n' \
                 '|  \\\\    /   O peration     | Version:  2.0.0                                 |\n' \
                 '|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |\n' \
                 '|    \\\\/     M anipulation  |                                                 |\n' \
                 '\\*---------------------------------------------------------------------------*/\n' \
                 'FoamFile\n' \
                 '{\n' \
                 '    version     2.0;\n' \
                 '    format      ascii;\n' \
                 '    class       dictionary;\n' \
                 '    object      blockMeshDict;\n' \
                 '}\n' \
                 '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n')

  def writeConversionFactor(self):
    self.f.write('convertToMeters {0:0.3f};\n\n'.format(self.conversionFactor))

  def writePoints(self):
    self.f.write('vertices\n(\n')
    for z in [0,1]:
      for i in range(len(self.point)):
        self.f.write('{0}({1:10.6f} {2:10.6f} {3:3.1f})\n'.format(' '*4, self.point[i].pos[0], \
                                                                         self.point[i].pos[1], z))
    self.f.write(');\n\n')

  def writeBlocks(self):
    self.f.write('blocks\n(\n')
    for i in range(len(self.block)):
      self.f.write('{0}hex ('.format(' '*4))
      for j in range(len(self.block[i].ptInd)):
        self.f.write('{0:2d} '.format(self.block[i].ptInd[j]))
      self.f.write(') ({0:3d} {1:3d} 1) '.format(self.block[i].nPts[0], \
                                                 self.block[i].nPts[1]))
      self.f.write('edgeGrading (')
      for j in range(len(self.block[i].grading)):
        self.f.write('{0:6.3f} '.format(self.block[i].grading[j]))
      self.f.write('{0})\n'.format('1 '*4))
    self.f.write(');\n\n')
                    
  def writeEdges(self):
    self.f.write('edges\n(\n')

    self.f.write(');\n\n')

  def writeBoundaries(self):
    self.f.write('boundary\n(\n')
    for i in range(len(self.boundary)):
      self.f.write('{0}{1}\n{2}{{\n'.format(' '*4, self.boundary[i].name, ' '*4))
      self.f.write('{0}type {1};\n'.format(' '*8, self.boundary[i].patchType))
      self.f.write('{0}faces\n{1}(\n'.format(' '*8, ' '*8))
      for j in range(len(self.boundary[i].facePts)):
        self.f.write('{0}('.format(' '*12))
        for k in range(len(self.boundary[i].facePts[j])):
          self.f.write('{0:3d} '.format(self.boundary[i].facePts[j][k]))
        self.f.write(')\n')
      self.f.write('{0});\n{1}}}\n\n'.format(' '*8, ' '*4))
    self.f.write(');\n\n')

  def writeClosing(self):
    self.f.write('mergePatchPairs\n(\n);\n\n')
    self.f.write('// ************************************************************************* //\n')

class Point:

  def __init__(self, ID, ind):
    self.ID  = ID
    self.ind = ind
    self.pos = np.zeros(2)

  def plot(self):
    plt.plot(self.pos[0], self.pos[1], 'r*')


class Line:

  def __init__(self, ID, ind, pt1, pt2, grading):
    self.ID = ID
    self.ind = ind
    self.ptID = [pt1, pt2]
    self.grading = grading
    self.pts = []
    self.ptInd = []

  def plot(self):
    x = []
    y = []
    for i in range(len(self.pts)):
      x.append(self.pts[i].pos[0])
      y.append(self.pts[i].pos[1])
    plt.plot(x, y, 'b-')


class Block:

  def __init__(self, name, ind, lineID):
    self.name = name
    self.ind = ind
    self.lineID = lineID
    self.lineInd = []
    self.lines = []
    self.baseCorner = 0
    self.nPts = []
    self.pts = []
    self.ptID = []
    self.ptInd = []
    self.grading = []

  def plot(self):
    #for i in [0,1,3,2,0,3,2,1]:
    #  x.append(self.pts[i].pos[0])
    #  y.append(self.pts[i].pos[1])
    for i in [0,1]:
      x = []
      y = []
      for j in [0,2]:
        x.append(self.pts[i+j].pos[0])
        y.append(self.pts[i+j].pos[1])
      plt.plot(x, y, 'm--')


class Boundary:

  def __init__(self, name, ind, patchType, lineID):
    self.name = name
    self.ind = ind
    self.patchType = patchType
    self.lineID = lineID
    self.lineInd = []
    self.lines = []
    self.facePts = []


def sind(ang):
  return np.sin(deg2rad(ang))
def cosd(ang):
  return np.cos(deg2rad(ang))
def tand(ang):
  return sind(ang)/cosd(ang)
def deg2rad(ang):
  return ang * np.pi / 180
def ang2vec(ang):
  return np.array([cosd(ang), sind(ang)])
