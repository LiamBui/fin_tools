import math

def funds_round(num, places=2, direction=math.floor):
    return direction(num * (10**places)) / float(10**places)

def update_avg_entry(avg_entry, available_btc, buy_price, buy_size):
	return (avg_entry * available_btc + (buy_size * buy_price)) / (available_btc + buy_size)

def update_pnl(pnl, avg_entry, sell_price, sell_size):
	return pnl + sell_size * (sell_price - avg_entry)