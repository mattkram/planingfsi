import os
import re
import json

import numpy

from planingfsi import unit

class Dictionary(dict):

    def __init__(self, fromFile=None, fromString=None):
        # Keep reference to Dict for backwards compatibility
        self.Dict = self
        if fromFile is not None and os.path.exists(fromFile):
            self.loadFromFile(fromFile)
        elif fromString is not None:
            self.loadFromString(fromString)
        
        # If specified, read values from a base dictionary
        # All local values override the base dictionary values
        baseDictDir = self.readOrDefault('baseDict', None)
        if baseDictDir is not None:
            # Allow for loading dictionary from different directory by tracing relative references from original file directory
            if baseDictDir.startswith('..'):
                baseDictDir = os.path.join(os.path.dirname(fromFile), baseDictDir)
            baseDict = Dictionary(baseDictDir)
            for key in baseDict.Dict:
                if not key in self:#.Dict:
                    self.update(key, baseDict[key])

    def jsonifyString(self, string):
        # Loop through a string, ensuring double-quotes are used to comply with json standard. 
        # Find pattern.
        # Add everything up until the pattern to new copy of string.
        # Add pattern with single-quotes substituted for double-quotes.
        # If there's a double-quote inside single-quotes, it means we have an apostrophe.
        # Add everything up to the double quote.
        # Once they are all added, which will eventually include the apostrophe, normal matching will proceed.
        print(string)
        input()

        # Make sure string is surrounded by curly brackets
        if not string.startswith('{'):
            string = ''.join(('{', string, '}'))
         
        # Replace all single-quoted strings with double-quotes
        pattern = re.compile("'(.+?)'")
        newString = ''
        while True:
            # Find first match of pattern in string
            match = re.search(pattern, string)
            if not match:
                break
    
            group = match.group(0)
            
            if not '"' in group:
                removeInd = match.end(0)
                # Add everything to end of group to new string, with single quotes replaced with double
                addString = ''.join((string[:match.start(0)], group.replace("'", '"')))
            else:
                removeInd = match.start(0) + group.find('"')
                addString = string[:removeInd]
            
            newString = ''.join((newString, addString))
            string = string[removeInd:]

        # Add anything remaining after last match
        newString = ''.join((newString, string))

        # Replace environment variables with their value
        def repl(m):
            return os.environ[m.group(1)]
        newString = re.sub('\$(\w+)', repl, newString)

        # Find all words that aren't numbers. If they don't have quotes add them.
        wordPattern = re.compile(r"[^:,\{\}\[\]]+")
        stringPattern = re.compile(r"[A-Za-z+-].*\w|\w.*[A-Za-z\"]|[\/\%].*")
        currInd = 0
        while True:
            wordMatch = re.search(wordPattern, newString[currInd:])
            if not wordMatch:
                break
            word = wordMatch.group(0).strip()
            print(word)
            stringMatch = re.match(stringPattern, word)
            if stringMatch:
                newString = ''.join((newString[:currInd+wordMatch.start()], 
                                     '"', word, '"', newString[currInd+wordMatch.end():]))
                currInd += wordMatch.end() + 2
            else:
                currInd += wordMatch.end()
        
        return newString

    def loadFromString(self, string):
        string = self.jsonifyString(string)
        try:
            dict_ = json.loads(string)
        except:
            raise ValueError( 'Error converting string to json: {0}'.format(string))

        # Copy items from json dictionary to instance Dict attribute
        for key, val in dict_.iteritems():
            if isinstance(val, basestring):
                if re.match('[+-]?nan', val):
                    val = float('nan')
                elif re.match('[+-]?inf|[0-9]+[Ee][+-]?[0-9]+', val):
                    val = float(val) 
                # Evaluate if unit. is in the string
                elif 'unit.' in val:
                    try:
                        val = eval(val)
                    except: 
                        pass
                
            self.update(key, val)


    def loadFromFile(self, filename):
        # Convert file format to appropriate string, then load dict from the string
        dictList = []
        with open(filename) as f:
            # Read file line-by-line until end of file
            while True:
                line = f.readline()
                if not line:
                    break
                
                # Remove comment strings, everything after # discarded
                line = line.split('#')[0]
                dictList.append(line.strip())
        
        # Join list with commas and replace potential substrings with extra commas
        # TODO This will break if multiple commas or brackets with commas exist inside quotes
        dictStr = ''.join(('{', ','.join(dictList), '}'))
        if dictStr.startswith('{{') and dictStr.endsWith('}}'):
            dictStr = dictStr[1:-1]
        dictStr = re.sub(',,+', ',', dictStr) 
        dictStr = dictStr.replace('[,', '[')
        dictStr = dictStr.replace('{,', '{')
        dictStr = dictStr.replace(',]', ']')
        dictStr = dictStr.replace(',}', '}')
        dictStr = dictStr.replace('}{', '},{')
        
        print(dictStr)
        input()

        self.loadFromString(dictStr)

    def removeQuotes(self, string):
        # Remove string quotes if they are included
        if ((string.startswith("'") and string.endswith("'")) or
            (string.startswith('"') and string.endswith('"'))):
            return string[1:-1]
        else:      
            return string
      
    def update(self, key, value):
        self[key] = value
      
    def read(self, key, default=None, dataType=None):
        val = self.readOrDefault(key, default)
        if dataType is not None:
            val = dataType(val)
        return val

    def readOrDefault(self, key, default):
        return self.get(key, default)

    def readAsList(self, key, dataType=str):
        # Read key as list. Will convert string to list with items of datatype specified if needed.
        string = self.read(key)
        if isinstance(string, basestring):
            string = string.replace('[','').replace(']','')
            
            return [dataType(self.removeQuotes(v.strip())) for v in string.split(',')]
        else:
            return string

    def readAsNumpyArray(self, key):
        return numpy.array(self.readAsList(key, float))

    def readAsArray(self, *args):
        return self.readAsNumpyArray(*args)
