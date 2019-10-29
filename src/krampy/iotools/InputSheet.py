import sys
import xlrd
import krampy as kp

# A class to automatically and flexibly collect the data from a row of an input excel sheet
class InputRow:
    def storeValue(self, varName, value):

        if value == "":
            value = None
        elif isinstance(value, str):  # one-length unicode strings
            if value.lower() == "y":
                value = True
            elif value.lower() == "n":
                value = False
        setattr(self, varName, value)


class InputSheet:
    def __init__(self, bookName, sheetName, varList, **kwargs):
        if isinstance(varList, dict):
            self.varList = varList
        else:
            self.varList = {}
            for var in varList:
                if (isinstance(var, tuple) or isinstance(var, list)) and len(var) == 3:
                    self.varList[var[0]] = (var[1], var[2])
                else:
                    self.varList[var[0]] = var[1]
        self.numHeaderRows = kwargs.get(
            "num_header_rows", kwargs.get("numHeaderRows", 1)
        )
        self.headerMatchRow = kwargs.get(
            "header_match_row", kwargs.get("headerMatchRow", 0)
        )
        self.rowList = []
        self.numRows = 0
        self.defaultValue = kwargs.get("default", None)

        self.loadSheet(bookName, sheetName)

    def __iter__(self):
        for row in self.rowList:
            yield row

    def __getitem__(self, index):
        return self.rowList[index]

    def getRowByMatch(self, attributeName, value):
        return self[kp.match(self.getList(attributeName), value)]

    def getRowValue(self, indexVar, matchVar, matchValue):
        return kp.indexMatch(self.getList(indexVar), self.getList(matchVar), matchValue)

    def getList(self, varName):
        return [getattr(row, varName) for row in self.rowList]

    def getColNum(self, wildcard, default=None):
        try:
            return kp.match(self.header, wildcard)
        except (ValueError, IndexError):
            if default is not None:
                return default
            else:
                print "Could not match wildcard {0} to a column".format(wildcard)
                return None

    def loadSheet(self, bookName, sheetName):
        if isinstance(bookName, xlrd.Book):
            book = bookName
        else:
            book = xlrd.open_workbook(bookName)

        if sheetName in book.sheet_names():
            sheet = book.sheet_by_name(sheetName)

            # Store header
            self.header = sheet.row_values(self.headerMatchRow)

            # Create a temporary dictionary where each entry is a column
            self.numRows = sheet.nrows - self.numHeaderRows
            tmpDict = {}
            for key, val in self.varList.iteritems():
                if isinstance(val, tuple):
                    val = val[0]
                colNum = self.getColNum(val, default=sys.maxint)
                if colNum is not None and colNum < sys.maxint:
                    tmpDict[key] = sheet.col_values(
                        colNum, start_rowx=self.numHeaderRows
                    )
                else:
                    tmpDict[key] = [self.defaultValue for _ in range(self.numRows)]

            # Add rows to row list
            for i in range(self.numRows):
                row = InputRow()
                for key, val in self.varList.iteritems():
                    if isinstance(val, tuple):
                        row.storeValue(key, map(val[1], [tmpDict[key][i]])[0])
                    else:
                        row.storeValue(key, tmpDict[key][i])
                self.rowList.append(row)
        else:
            print 'Sheet "{0}" does not exist in workbook "{1}". Skipping import.'.format(
                sheetName, bookName
            )
