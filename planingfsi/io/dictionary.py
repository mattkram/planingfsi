import os
import re
import json

import numpy

from planingfsi import unit

class Pattern(object):
    DELIMITER = re.compile(r'[:,\{\}\[\]]')
    NUMBER = re.compile(r'\A[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\Z')
    LITERAL = re.compile(r'(\"[\w\\\/\.]*\")|(\'[\w\\\/\.]*\')')
    BOOL = re.compile(r'(True|true|False|false)')
    NONE = re.compile(r'(None|none)')
    WORD = re.compile(r'[-+\w]+')
    ENV = re.compile(r'\$(\w+)')
    NANINF = re.compile(r'[+-]?(nan|inf)')
    ALL = (DELIMITER, NUMBER, LITERAL, BOOL, WORD)


class Dictionary(dict):

    def __init__(self, fromFile=None, fromString=None, from_dict=None):
        # Keep reference to Dict for backwards compatibility
#        self.Dict = self
        if fromFile is not None and os.path.exists(fromFile):
            self.loadFromFile(fromFile)
        elif fromString is not None:
            self.loadFromString(fromString)
        elif from_dict is not None:
            self.loadFromDict(from_dict)
        
        # If specified, read values from a base dictionary
        # All local values override the base dictionary values
        baseDictDir = self.read('baseDict', default=None)
        if baseDictDir is not None:
            # Allow for loading dictionary from different directory by tracing relative references from original file directory
            if baseDictDir.startswith('..'):
                baseDictDir = os.path.join(os.path.dirname(fromFile), baseDictDir)
            baseDict = Dictionary(baseDictDir)
            for key, val in baseDict.items():
                if not key in self:
                    self.update(key, val)

    def loadFromFile(self, filename):
        # Convert file format to appropriate string, then load dict from the string
        dictList = []
        with open(filename) as f:
            for line in f:                
                # Remove comment strings, everything after # discarded
                line = line.split('#')[0].strip()
                if line != '':
                    dictList.append(line)
        
        dictStr = ','.join(dictList)
        self.loadFromString(dictStr)

    def loadFromString(self, inString):
        # Convert a string to a json-compatible string
        
        # Surround string with curly brackets if it's not already
        if not inString.startswith('{'):
            inString = inString.join('{}')
         
        # Replace multiple consecutive commas with one
        inString = re.sub(',,+', ',', inString) 
        
        def repl(m):
            return m.group(1)
        
        # Replace [, or {, or ,] or ,} with bracket only
        inString = re.sub(r'([\{\[]),', repl, inString)
        inString = re.sub(r',([\]\}])', repl, inString)
        
        inString = inString.replace('}{', '},{')

        # Replace environment variables with their value
        def repl(m):
            return os.environ[m.group(1)].join('""')
        inString = Pattern.ENV.sub(repl, inString)

        # Split inString into a list of delimiters and sub-strings
        inList = []
        while(len(inString) > 0):
            inString = inString.strip()
            match = Pattern.DELIMITER.search(inString)
            if match:
                if match.start() > 0:
                    word = inString[:match.start()].strip()
                    if Pattern.LITERAL.match(word):
                        # Surround literals by escaped double-quotes
                        word = word[1:-1].join(('\\"', '\\"')).join('""')
                    elif Pattern.BOOL.match(word):
                        # Convert boolean values to lowercase
                        word = word.lower()
                    elif Pattern.NUMBER.match(word):
                        pass
                    elif Pattern.WORD.match(word):
                        # Surround words by double-quotes
                        word = word.join('""')
                    else:
                        word = word.join(('\\"', '\\"')).join('""')

                    inList.append(word)

                inList.append(match.group(0))
                inString = inString[match.end():]
            else:
                break
        jsonString = ''.join(inList)       
        self.loadFromJson(jsonString)

    def loadFromJson(self, string):
        try:
            dict_ = json.loads(string)
        except:
            raise ValueError( 'Error converting string to json: {0}'.format(string))
        
        self.loadFromDict(dict_)

    def loadFromDict(self, dict_):
        # Copy items from json dictionary to self, with some special processing
        for key, val in dict_.items():
            if isinstance(val, str):
                # Process NaN and infinity
                match = Pattern.NANINF.match(val)
                if match:
                    val = float(match.group(0))
                elif Pattern.LITERAL.match(val):
                    # Remove quotes from literal string
                    val = val[1:-1]
                elif Pattern.NONE.match(val):
                    # Convert None string to None
                    val = None
                # Evaluate if unit. is in the string
                elif 'unit.' in val:
                    try:
                        val = eval(val)
                    except: 
                        ValueError('Cannot process the value {0}: {1}'.format(key, val))

            elif isinstance(val, dict):
                val = Dictionary(from_dict=val)
           
            # Remove quotes from key
            if Pattern.LITERAL.match(key):
                key = key[1:-1]

            self.update(key, val)

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

    def readLoadOrDefault(self, key, default):
        val = self.read(key, default)
        if isinstance(val, str):
            # Here because if in preamble there is
            # a circular import
            from planingfsi import config
            val = getattr(config, val)
        return val

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
