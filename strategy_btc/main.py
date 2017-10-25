from strategies.reverse_bollinger_band import ReverseExpStrategy
from strategies.reverse_bollinger_double_bands import ReverseExpDualBandStrategy
from TradeSimulator import TradeSimulator
from EnvironmentSimulator import EnvironmentSimulator
from analyzer import Analyzer
import numpy as np

'''
All locations commented out for reference.
'''
# LOCATIONS = ['C:/Users/lb/fin_tools/data/btc/20170511-20170521.log','C:/Users/lb/fin_tools/data/btc/20170721-20170729.log',
#              'C:/Users/lb/fin_tools/data/btc/20170730-20170807.log','C:/Users/lb/fin_tools/data/btc/20170808-20170815.log',
#              'C:/Users/lb/fin_tools/data/btc/20170816-20170823.log','C:/Users/lb/fin_tools/data/btc/20170824-20170901.log',
#              'C:/Users/lb/fin_tools/data/btc/20170902-20170911.log','C:/Users/lb/fin_tools/data/btc/20170912-20170918.log',
#              'C:/Users/lb/fin_tools/data/btc/20170919-20170925.log','C:/Users/lb/fin_tools/data/btc/20170926-20171002.log',
#              'C:/Users/lb/fin_tools/data/btc/20171003-20171009.log','C:/Users/lb/fin_tools/data/btc/20171010-20171016.log',
#              'C:/Users/lb/fin_tools/data/btc/20171017-20171021.log']

LOCATIONS = ['C:/Users/lb/fin_tools/data/btc/20170824-20170901.log','C:/Users/lb/fin_tools/data/btc/20170902-20170911.log',
             'C:/Users/lb/fin_tools/data/btc/20170912-20170918.log','C:/Users/lb/fin_tools/data/btc/20170919-20170925.log',
             'C:/Users/lb/fin_tools/data/btc/20170926-20171002.log','C:/Users/lb/fin_tools/data/btc/20171003-20171009.log',
             'C:/Users/lb/fin_tools/data/btc/20171010-20171016.log','C:/Users/lb/fin_tools/data/btc/20171017-20171021.log']


ENVIRONMENT = EnvironmentSimulator(locations=LOCATIONS, funds=100.0)
TRADER = TradeSimulator(slippage_type='static', slippage_params={'p': 0.005})

STRATEGY = ReverseExpDualBandStrategy(ENVIRONMENT, TRADER, epsilon_upper=0.6, epsilon_lower=-0.6, halflife=720, halflife2=1560)
ANALYZER = Analyzer(ENVIRONMENT, TRADER, STRATEGY, iterations=5)
ANALYZER.run()