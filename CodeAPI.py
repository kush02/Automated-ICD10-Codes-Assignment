import sys
sys.dont_write_bytecode = True

import requests
import json
import argparse


baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
efetch = "efetch.fcgi?db=pubmed"
ids = "&id=16778410"
options = "&retmode=text&rettype=medline"

queryURL = baseURL + efetch + ids + options
r = requests.get(queryURL)
#print(r.text)

