import os
import sys


PROJECT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(PROJECT_DIR, 'tests')

sys.path.insert(0, PROJECT_DIR)
