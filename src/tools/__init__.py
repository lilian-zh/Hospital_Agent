'''
Adapted from AIDOC
'''

from .tool_execs import *

TOOLS = [
    request_physical_exam,
    request_blood_test, 
    request_urine_test,
    request_radiology,
    request_microbiology,
    prescribe_medication,
    search_procedure,
    request_procedure,
    finish,
         ]