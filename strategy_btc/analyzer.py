import pandas as pd
import numpy as np
from shared import get_data

class Analyzer(object):

	def __init__(self, environment, trader, strategy, iterations=1):

		self.environment = environment
		self.trader = trader
		self.strategy = strategy
		self.iterations = iterations

	def find_total_days(self):
		
		total_days = 0

		for location in self.environment.locations:
			df = get_data(location)
			num_days = (df.iloc[len(df.index)-1].name - df.iloc[0].name).days
			total_days += num_days

		return total_days

	def run(self):

		self.environment.reinit()

		self.overall_days = self.find_total_days()
		self.overall_returns = []
		self.overall_prob_win = []
		self.overall_prob_loss = []
		self.overall_avg_win = []
		self.overall_avg_loss = []
		self.overall_ev = []
		self.overall_len = []
		self.overall_max_wins = []
		self.overall_min_losses = []
		self.overall_med_wins = []
		self.overall_med_losses = []
		self.overall_max_dds = []
		self.peak = None
		self.trough = None

		for i in range(self.iterations):

			self.environment.reinit()

			self.strategy.run()

			diffs = np.diff(self.environment.pnls)
			wins = filter(lambda x: x > 0, diffs)
			losses = filter(lambda x: x < 0, diffs)
			
			self.overall_returns.append(self.environment.pnl)

			if len(wins) + len(losses) == 0:
				prob_win = 0
				prob_loss = 0
			else:
				prob_win = len(wins) / float(len(wins) + len(losses))
				prob_loss = len(losses) / float(len(wins) + len(losses))

			if len(wins) == 0:
				avg_win = 0
			else:
				avg_win = float(sum(wins)) / len(wins)

			if len(losses) == 0:
				avg_loss = 0
			else:
				avg_loss = float(sum(losses)) / len(losses)

			self.overall_prob_win.append(prob_win)
			self.overall_prob_loss.append(prob_loss)
			self.overall_avg_win.append(avg_win)
			self.overall_avg_loss.append(avg_loss)
			self.overall_ev.append(prob_win * avg_win + prob_loss * avg_loss)
			self.overall_len.append(len(self.environment.pnls))
			self.overall_max_wins.append(max(wins))
			self.overall_min_losses.append(min(losses))
			self.overall_med_wins.append(np.median(wins))
			self.overall_med_losses.append(np.median(losses))

			i = np.argmax(np.maximum.accumulate(self.environment.returns) - self.environment.returns) # end of max dd period
			j = np.argmax(self.environment.returns[:i]) # start of max dd period
			peak = self.environment.returns[j]
			trough = self.environment.returns[i]
			dd = (trough - peak) / peak
			self.overall_max_dds.append(dd)
			if dd >= np.max(self.overall_max_dds):
				sells = filter(lambda x: x['size'] < 0, self.environment.trades)
				self.peak = sells[j]['time']
				self.trough = sells[i-1]['time']

			self.environment.reinit()

		avg_return = float(sum(self.overall_returns)) / len(self.overall_returns)
		avg_prob_win = float(sum(self.overall_prob_win)) / len(self.overall_prob_win)
		avg_prob_loss = float(sum(self.overall_prob_loss)) / len(self.overall_prob_loss)
		avg_win_amt = float(sum(self.overall_avg_win)) / len(self.overall_avg_win)
		avg_loss_amt = float(sum(self.overall_avg_loss)) / len(self.overall_avg_loss)
		avg_num_trades = float(sum(self.overall_len)) / len(self.overall_len)
		avg_ev = float(sum(self.overall_ev)) / len(self.overall_ev)
		avg_max_win = float(sum(self.overall_max_wins)) / len(self.overall_max_wins)
		avg_min_loss = float(sum(self.overall_min_losses)) / len(self.overall_min_losses)
		med_win = np.median(self.overall_med_wins)
		med_loss = np.median(self.overall_med_losses)
		avg_max_dd = float(sum(self.overall_max_dds)) / len(self.overall_max_dds)

		print '\t \t --------- Analysis Report ---------'
		print str(self.strategy)
		print '\t ---------------------------------------------'
		print 'Total number of days analyzed: \t {}'.format(self.overall_days)
		print 'Average number of trades: \t {:.2f}'.format(avg_num_trades)
		print 'Average return: \t {:.2f}'.format(avg_return)
		print 'Median return: \t {:.2f}'.format(np.median(self.overall_returns))
		print 'Standard deviation of returns: \t {:.2f}'.format(np.std(self.overall_returns))
		print 'Average win probability: \t {:.2f}'.format(avg_prob_win)
		print 'Average loss probability: \t {:.2f}'.format(avg_prob_loss)
		print 'Median win probability: \t {:.2f}'.format(np.median(self.overall_prob_win))
		print 'Median loss probability: \t {:.2f}'.format(np.median(self.overall_prob_loss))
		print 'Average win amount: \t {:.2f}'.format(avg_win_amt)
		print 'Average loss amount: \t {:.2f}'.format(avg_loss_amt)
		print 'Median win amount: \t {:.2f}'.format(med_win)
		print 'Median loss amount: \t {:.2f}'.format(med_loss)
		print 'Average highest win: \t {:.2f}'.format(avg_max_win)
		print 'Average lowest loss: \t {:.2f}'.format(avg_min_loss)
		print 'Median highest win: \t {:.2f}'.format(np.median(self.overall_max_wins))
		print 'Median lowest loss: \t {:.2f}'.format(np.median(self.overall_min_losses))
		print 'Average expected value: \t {:.2f}'.format(avg_ev)
		print 'Median expected value: \t {:.2f}'.format(self.overall_ev)
		print 'Average maximum drawdown: \t {:.2f}'.format(avg_max_dd)
		print 'Median maximum drawdown: \t {:.2f}'.format(np.median(self.overall_max_dds))
		print 'Time of maximum drawdown: \t {} to {}'.format(self.peak, self.trough)
		print ''
