import csv
import os
import sys
import xlwt
import xlrd
import xlutils.copy as xlcp

import krampy as kp


class OutputWorkbook:
    def __init__(self, name, unitDict=None):
        self.name = name
        if os.path.exists(name):
            self.xlObj = xlrd.open_workbook(name)
            self.sheetNames = self.xlObj.sheet_names()
            self.sheetNumCells = [(s.nrows, s.ncols) for s in self.xlObj.sheets()]
            self.xlObj = xlcp.copy(self.xlObj)
        else:
            self.sheetNames = []
            self.xlObj = xlwt.Workbook()

        self.sheets = []
        self.unitDict = unitDict

    def addSheet(self, sheetName):
        sheet = OutputWorksheet(self, sheetName, self.unitDict)
        self.sheets.append(sheet)
        return sheet

    def save(self):
        try:
            for i in range(10):
                if i == 0:
                    newFileName = self.name
                else:
                    newFileName = "{0}_{1}.xls".format(self.name.replace(".xls", ""), i)

                try:
                    self.xlObj.save(newFileName)
                    print("Output workbook saved to {0}".format(newFileName))
                    return None
                except:
                    i += 0
        except:
            raise NameError("Cannot find an appropriate output file path.")


class OutputWorksheet:
    def __init__(self, parent, name, unitDict=None):
        if not unitDict:
            unitDict = {"nd": "-"}

        self.name = name
        # If sheet already exists, load it, otherwise create it
        if self.name in parent.sheetNames:
            ind = parent.sheetNames.index(self.name)
            self.xlObj = parent.xlObj.get_sheet(ind)
            for i in range(parent.sheetNumCells[ind][0]):
                for j in range(parent.sheetNumCells[ind][1]):
                    self.xlObj.write(i, j, "")
        else:
            self.xlObj = parent.xlObj.add_sheet(self.name)
        self.activeRow = 0
        self.unitDict = unitDict

    def getHeaderString(self, header):
        if not isinstance(header, str):
            name = header[0]
            unit = header[1]
        else:
            name = header
            unit = "nd"

        return "{0} [{1}]".format(name, self.unitDict.get(unit, unit))

    def writeHeaders(self, *args):
        self.writeHeader(*args)

    def writeHeader(self, *args):
        self.writeRow(*[self.getHeaderString(arg) for arg in args])

    def writeRow(self, *values):
        # Write values to the next row
        activeCol = 0
        for value in values:
            if not hasattr(value, "__iter__"):
                value = [value]
            for valuei in value:
                self.xlObj.write(self.activeRow, activeCol, valuei)
                activeCol += 1
        self.activeRow += 1

    def writeCell(self, indx, indy, value):
        self.xlObj.write(indx, indy, value)


class OutputCSV:
    def __init__(self, name, unitDict=None):
        self.name = name
        self.fObj = open(name, "wb")
        self.writer = csv.writer(self.fObj, delimiter=",")
        self.activeRow = 0

        if not unitDict:
            self.unitDict = {"nd": "-"}
        else:
            self.unitDict = unitDict

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        return isinstance(value, TypeError)

    def save(self):
        self.close()

    def close(self):
        self.fObj.close()

    def getHeaderString(self, header):
        if not isinstance(header, str):
            name = header[0]
            unit = header[1]
        else:
            name = header
            unit = "nd"

        return "{0} [{1}]".format(name, self.unitDict.get(unit, unit))

    def writeHeaders(self, *args):
        self.writeHeader(*args)

    def writeHeader(self, *args):
        self.writeRow(*[self.getHeaderString(arg) for arg in args])

    def writeRow(self, *values):
        # Concatenate all arguments into one list and write row to file
        # Stacks each value as if they were lists combined
        writeList = []
        for value in values:
            if hasattr(value, "__iter__"):
                writeList += [v for v in value]
            else:
                writeList += [value]
        self.writer.writerow(writeList)


class CombinedOut:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)


class LogFile:
    def __init__(self, *fileNames, **kwargs):
        dirName = kwargs.get("dirName", ".")
        kp.mkdir_p(dirName)
        self.files = [open(os.path.join(dirName, ff), "w") for ff in fileNames]
        sys.stdout = CombinedOut(sys.stdout, *self.files)
        sys.stderr = CombinedOut(sys.stderr, *self.files)

    def close(self):
        for ff in self.files:
            ff.close()


class TimestampedLogFile(LogFile):
    def __init__(self, baseName, startTime, **kwargs):
        timeString = startTime.replace(microsecond=0).isoformat().replace(":", "-")
        LogFile.__init__(self, "{0}_{1}.dat".format(baseName, timeString), **kwargs)


def writeRowToFile(f, *values):
    # Write values to the next row
    activeCol = 0
    for i, value in enumerate(values):
        if not hasattr(value, "__iter__"):
            value = [value]
        for j, valuei in enumerate(value):
            if i > 0 or j > 0:
                f.write(", ")
            f.write("{0}".format(valuei))
            activeCol += 1
    f.write("\n")


def getLineCSV(f):
    return [float(s.replace(" ", "")) for s in f.readline().split(",")]


def _getOutCell(outSheet, rowIndex, colIndex):
    """ HACK: Extract the internal xlwt cell representation. """
    row = outSheet._Worksheet__rows.get(rowIndex)
    if not row:
        return None

    cell = row._Row__cells.get(colIndex)
    return cell


def setOutCell(outSheet, row, col, value):
    """ Change cell value without changing formatting. """
    previousCell = _getOutCell(outSheet, row, col)

    outSheet.write(row, col, value)

    if previousCell:
        newCell = _getOutCell(outSheet, row, col)
        if newCell:
            newCell.xf_idx = previousCell.xf_idx


# Thread-safe printing
def safePrint(s=""):
    print("{0}\n".format(s))


# Print functions for various sections, etc.
HEADER_WIDTH = 100
TEXT_WIDTH = HEADER_WIDTH - 6


def printHeaderLine(s=""):
    if len(s) > TEXT_WIDTH:
        printHeaderLine(s[:TEXT_WIDTH])
        printHeaderLine(s[TEXT_WIDTH:])
    else:
        safePrint("|| {0:{1}s} ||".format(s, TEXT_WIDTH))


def printSpacerLine():
    safePrint("=" * HEADER_WIDTH)


def printSubsection(s):
    safePrint()
    printSpacerLine()
    safePrint(s)
    printSpacerLine()
    safePrint()


def printHeader(*args):
    """
      Print header. Each argument will be printed on its own line. A value of None
      provided will be printed as a blank line. A spacer line is printed before and
      after the provided arguments.
    """
    printSpacerLine()
    for arg in args:
        if arg is not None:
            printHeaderLine(arg)
        else:
            printHeaderLine()
    printSpacerLine()


# Function to convert a list to a string (for printing)
def list2str(l, formatString=""):
    s = "["
    for i, li in enumerate(l):
        s += "{0:{1}}".format(li, formatString)
        if i < len(l) - 1:
            s += ", "
    s += "]"
    return s


# def selectFile(*filetypes, **kwargs):
#     from Tkinter import Tk
#     from tkFileDialog import askopenfilename, askopenfilenames
#
#     title = kwargs.get('title', 'Please select file(s)')
#     root = Tk()
#     root.withdraw()
#     filename = askopenfilename(title=title,
#                 filetypes=[filetype for filetype in filetypes]+[('All files', '*.*')])
#
#     # Error if no file selected
#     if len(filename) == 0:
#         raise NameError('Please select one or more appropriate files')
#
#     return filename
#
# def selectFiles(*filetypes, **kwargs):
#     from Tkinter import Tk
#     from tkFileDialog import askopenfilename, askopenfilenames
#
#     title = kwargs.get('title', 'Please select file(s)')
#     root = Tk()
#     root.withdraw()
#     filenames = root.tk.splitlist(askopenfilenames(title=title,
#                 filetypes=[filetype for filetype in filetypes]+[('All files', '*.*')]))
#
#     # Error if no file selected
#     if len(filenames) == 0:
#         raise NameError('Please select one or more appropriate files')
#     return filenames
