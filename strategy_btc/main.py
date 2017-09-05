from strategies.reverse_bollinger_band import ReverseExpStrategy
from TradeSimulator import TradeSimulator
from EnvironmentSimulator import EnvironmentSimulator
from analyzer import Analyzer

LOCATIONS = ['C:/Users/lb/fin_tools/data/btc/20170511-20170521.log','C:/Users/lb/fin_tools/data/btc/20170721-20170729.log',
             'C:/Users/lb/fin_tools/data/btc/20170730-20170807.log','C:/Users/lb/fin_tools/data/btc/20170808-20170815.log',
             'C:/Users/lb/fin_tools/data/btc/20170816-20170823.log','C:/Users/lb/fin_tools/data/btc/20170824-20170901.log']
             
ENVIRONMENT = EnvironmentSimulator(locations=LOCATIONS, funds=100.0)
TRADER = TradeSimulator(slippage_type='static', slippage_params={'p': 0.01})
STRATEGY = ReverseExpStrategy(ENVIRONMENT, TRADER)
ANALYZER = Analyzer(ENVIRONMENT, TRADER, STRATEGY, iterations=3)

ANALYZER.run()