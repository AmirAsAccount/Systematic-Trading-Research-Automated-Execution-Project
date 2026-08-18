
import ccxt
import pandas
import numpy
import requests
import time
import logging



def get_all_symbols(exchange_id: str) -> list[str]: