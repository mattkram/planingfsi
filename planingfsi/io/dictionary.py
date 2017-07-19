import os
import re
import json

import numpy

from planingfsi import unit

class Pattern(object):
    """A small helper object for storing Regex patterns.
    
    Class Attributes
    ----------------
    DELIMITER : SRE_Pattern
        Any single character of the following set (in parens): (:,{}[])
    NUMBER : SRE_Pattern
        Any number, including int, float, or exponential patterns.
        Will only match if entire string is number.
    LITERAL : SRE_Pattern
        Any normal chars or path delimiters (/\.) surrounded by single- or double-quotes
    BOOL : SRE_Pattern
        Case-insensitive boolean values, True or False
    NONE : SRE_Pattern
        Case-insensitive None
    WORD : SRE_Pattern
        A continuous string of normal chars, including plus and minus signs
    ENV : SRE_Pattern
        A continuous string of normal chars preceded by a dollar sign, $
    NANINF : SRE_Pattern
        Case-insensitive NaN or Inf, optionally with sign

    Usage
    -----
    Pattern.NUMBER.match('1.455e+10') will return a re.MatchObject

    """
    
    DELIMITER = re.compile(r'[:,\{\}\[\]]')
    NUMBER = re.compile(r'\A[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\Z')
    LITERAL = re.compile(r'(\"[\w\\\/\.]*\")|(\'[\w\\\/\.]*\')')
    BOOL = re.compile(r'(True|False)', re.IGNORECASE)
    NONE = re.compile(r'(None)', re.IGNORECASE)
    WORD = re.compile(r'[-+\w]+')
    ENV = re.compile(r'\$(\w+)')
    NANINF = re.compile(r'[+-]?(nan|inf)', re.IGNORECASE)
    ALL = (DELIMITER, NUMBER, LITERAL, BOOL, WORD)


class Dictionary(dict):
    """An file-based extension of the standard dict object
    
    Dictionaries can be easily read from files, which may also depend themselves
    on recursive loading of sub-Dictionaries.
    
    Parameters
    ----------
    construct_from : str, dict, or None, optional, default=None
        If a dict is provided, the values are copied into this instance.

        If a string is provided which starts with a curly bracket "{", the 
        string is parsed via the load_from_string method, which aims to be
        "smart" in cleaning the string into json format and subsequently 
        loading it as a dictionary.

        If the string does not begin with a curly bracket, it is taken to be
        a filename, which is in-turn loaded as a string and passed to the
        load_from_string method.

        If no value or None is provided, a blank dictionary is instantiated.
    
    """

    def __init__(self, construct_from=None):
        source_file = None

        # If no argument, just call default dict constructor
        if construct_from is None:
            super().__init__()
            return
        
        # Call appropriate function to load dictionary values
        if isinstance(construct_from, dict):
            self.load_from_dict(construct_from)
        elif isinstance(construct_from, str):
            if construct_from.strip().startswith('{'):
                self.load_from_string(construct_from)
            else:
                source_file = construct_from
                self.load_from_file(construct_from)

        else:
            raise ValueError('Argument to Dictionary must be string, dict, or None')

        # If specified, read values from a base dictionary
        # All local values override the base dictionary values
        base_dict_dir = self.read('baseDict', default=None)
        if base_dict_dir is not None:
            # Allow for loading dictionary from different directory by tracing
            # relative references from original file directory
            if base_dict_dir.startswith('..'):
                if source_file is not None:
                    # Begin from the directory the source_file lies in
                    dir_name = os.path.dirname(source_file)
                else:
                    # Otherwise, use the current directory
                    dir_name = '.'
                base_dict_dir = os.path.join(dir_name, base_dict_dir)
            base_dict = Dictionary(base_dict_dir)

            # Use values in base_dict if they don't exist in this Dictionary
            for key, val in base_dict.items():
                if not key in self:
                    self.update(key, val)

    def load_from_file(self, filename):
        """Load Dictionary information from a text file.

        Arguments
        ---------
        filename : str
            Name or path of file to load.
        """

        # Convert file format to appropriate string, then load dict from the string
        dict_list = []
        with open(filename) as f:
            for line in f:                
                # Remove comment strings, everything after # discarded
                line = line.split('#')[0].strip()
                if line != '':
                    dict_list.append(line)
        
        dict_string = ','.join(dict_list)
        self.load_from_string(dict_string)

    def load_from_string(self, inString):
        """Sequentially process runs in run list in either serial or parallel.
        
        Parameters
        ----------
        runList : list of BatchRunContainer, optional
            Optionally append a list of runs to the existing run list.
        
        """
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
        json_string = ''.join(inList)       
        self.load_from_json(json_string)

    def load_from_json(self, string):
        try:
            dict_ = json.loads(string)
        except:
            raise ValueError('Error converting string to json: {0}'.format(string))
        
        self.load_from_dict(dict_)

    def load_from_dict(self, dict_):
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
                val = Dictionary(val)
           
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
