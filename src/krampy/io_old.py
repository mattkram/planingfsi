"""A module holding a class to extend the built-in dictionary.

The Dictionary class is a subclass of the built-in dict type, and contains
methods for loading and instantiating the dictionary from text files.

Dictionary file import relies on converting the input file to json format,
then creating a dict object from the json string and performing some smart
type conversion for dictionary keys and values. The input dictionary should
be similar in format to a json dictionary, with the following allowable
modifications:
- Comma delimiters are not necessary between lines, as they will be automatically added
- The file does not need to begin and end with curly brackets
- Strings do not need to be surrounded by quotes, although they may be to
  force import as a string

The load methods us regular expression patterns defined in the Pattern class
to parse the tokens in the file. Patterns for numbers, words, None, NaN, Inf,
and environment variables exist.

"""
import json
import os
import re


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
        Any normal chars or path delimiters (/\.) surrounded by single- or
        double-quotes
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

    DELIMITER = re.compile(r"[:,\{\}\[\]]")
    NUMBER = re.compile(r"\A[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\Z")
    LITERAL = re.compile(r"(\"[\w\\\/\.]*\")|(\'[\w\\\/\.]*\')")
    BOOL = re.compile(r"(True|False)", re.IGNORECASE)
    NONE = re.compile(r"(None)", re.IGNORECASE)
    WORD = re.compile(r"[-+\w]+")
    ENV = re.compile(r"\$(\w+)")
    NANINF = re.compile(r"[+-]?(nan|inf)", re.IGNORECASE)


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
            if construct_from.strip().startswith("{"):
                self.load_from_string(construct_from)
            else:
                source_file = construct_from
                self.load_from_file(construct_from)

        else:
            raise ValueError("Argument to Dictionary must be string, dict, or None")

        # If specified, read values from a base dictionary
        # All local values override the base dictionary values
        base_dict_dir = self.read("baseDict", default=None)
        if base_dict_dir is not None:
            # Allow for loading dictionary from different directory by tracing
            # relative references from original file directory
            if base_dict_dir.startswith("."):
                if source_file is not None:
                    # Begin from the directory the source_file lies in
                    dir_name = os.path.dirname(source_file)
                else:
                    # Otherwise, use the current directory
                    dir_name = "."
                base_dict_dir = os.path.join(dir_name, base_dict_dir)
            base_dict = Dictionary(base_dict_dir)

            # Use values in base_dict if they don't exist in this Dictionary
            for key, val in base_dict.items():
                if not key in self:
                    self.update({key: val})

    def load_from_file(self, filename):
        """Load Dictionary information from a text file.

        Arguments
        ---------
        filename : str
            Name or path of file to load.
        """

        # Convert file format to appropriate string, then load dict from the string
        dict_list = []
        with open(filename) as ff:
            for line in ff:
                # Remove comment strings, everything after # discarded
                line = line.split("#")[0].strip()
                if line != "":
                    dict_list.append(line)

        dict_string = ",".join(dict_list)
        self.load_from_string(dict_string)

    def load_from_string(self, in_string):
        """Load Dictionary information from a string.

        Convert the string to a json-compatible string, then pass to
        load_from_json.

        Arguments
        ---------
        in_string : str
            String to parse.
        """

        # Surround string with curly brackets if it's not already
        if not in_string.startswith("{"):
            in_string = in_string.join("{}")

        # Replace multiple consecutive commas with one
        in_string = re.sub(",,+", ",", in_string)

        # Replace [, or {, or ,] or ,} with bracket only
        def repl(m):
            return m.group(1)

        in_string = re.sub(r"([\{\[]),", repl, in_string)
        in_string = re.sub(r",([\]\}])", repl, in_string)

        in_string = in_string.replace("}{", "},{")

        # Replace environment variables with their value & surround by quotes
        def repl(m):
            return os.environ[m.group(1)].join('""')

        in_string = Pattern.ENV.sub(repl, in_string)

        # Split in_string into a list of delimiters and sub-strings
        in_list = []
        while len(in_string) > 0:
            in_string = in_string.strip()
            match = Pattern.DELIMITER.search(in_string)
            if match:
                # Process any words before the delimiter
                if match.start() > 0:
                    word = in_string[: match.start()].strip()
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
                        # Surround by literal double-quotes, and regular ones
                        word = word.join(('\\"', '\\"')).join('""')

                    in_list.append(word)

                in_list.append(match.group(0))
                in_string = in_string[match.end() :]
            else:  # No more delimiters
                break
        json_string = "".join(in_list)
        self.load_from_json(json_string)

    def load_from_json(self, json_string):
        """Load Dictionary information from a json-formatted string.

        The string is first loaded by the json module as a dict, then passed to load_from_dict.

        Arguments
        ---------
        json_string : str
            String to parse.
        """
        try:
            dict_ = json.loads(json_string)
        except:
            raise ValueError("Error converting string to json: {0}".format(json_string))

        self.load_from_dict(dict_)

    def load_from_dict(self, dict_):
        """Load Dictionary information from a standard dict.

        The values are copied to this Dictionary and parsed, with special
        handling of certain variables.

        Arguments
        ---------
        dict_ : dict
            Dictionary to copy.
        """
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
                elif "unit." in val:
                    try:
                        val = eval(val)
                    except:
                        ValueError("Cannot process the value {0}: {1}".format(key, val))
            elif isinstance(val, dict):
                val = Dictionary(val)

            # Remove quotes from key
            if Pattern.LITERAL.match(key):
                key = key[1:-1]

            self.update({key: val})

    def read(self, key, default=None, type_=None):
        """Load value from dictionary.

        Arguments
        ---------
        key : str
            Dictionary key.
        default : optional, default=None
            Default value if key not in Dictionary.
        type_ : optional, default=No conversion
            Type to convert dictionary value to.
        """
        val = self.read_or_default(key, default)
        if type_ is not None:
            val = type_(val)
        return val

    def read_or_default(self, key, default):
        """Load value from dictionary, or return default if not found.

        Arguments
        ---------
        key : str
            Dictionary key.
        default :
            Default value if key not in Dictionary.
        """
        return self.get(key, default)

    def read_load_or_default(self, key, default):
        """Load value from dictionary. If value is a string, will load
        attribute of string name from config.

        Arguments
        ---------
        key : str
            Dictionary key.
        default :
            Default value if key not in Dictionary.
        """
        val = self.read(key, default)
        if isinstance(val, str):
            # Here because if in preamble there is
            # a circular import
            from planingfsi import config

            val = getattr(config, val)
        return val
