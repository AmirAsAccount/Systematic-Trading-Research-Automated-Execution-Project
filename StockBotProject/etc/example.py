#!/usr/bin/python3
# Get all available filter options
import csv
import pandas as pd
import finviz
from finviz.screener import Screener
filters = pd.DataFrame(Screener.load_filter_dict())
filters.to_csv('output.csv', index = False)  # ['Exchange', 'Index', 'Sector', 'Industry', ...]
