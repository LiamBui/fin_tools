import urllib2
import sys
import datetime

def date_strtofmt(start,end):
	startdate = datetime.datetime.strptime(start, '%m%d%Y')
	enddate = datetime.datetime.strptime(end, '%m%d%Y')
	start = startdate.strftime('%b+%d') + '%2C' + startdate.strftime('%Y')
	end = enddate.strftime('%b+%d') + '%2C' + enddate.strftime('%Y')
	return start, end

def construct_url(exchange,symbol,start,end):
	return 'https://www.google.com/finance/historical?q=' + exchange + '%3A' + symbol + '&startdate=' + start + '&enddate=' + end + '&output=csv'

def get_url(save_to, url):
	file = urllib2.urlopen(url)
	with open(save_to, 'wb') as output:
		output.write(file.read())

def main(argv):
	try:
		start,end = argv
	except:
		print "Please enter exactly two dates in the format of %m%d%Y. For exapmle, January 2, 2016 would be 01022016."
		sys.exit(2)
	if len(start) != 8 or len(end) != 8:
		print "Dates invalid. Please enter valid start and end destinations in the format of %m%d%Y. For example, January 1, 2016 would be 01022016."
		sys.exit(2)
	start,end = date_strtofmt(start,end)
	print 'Retrieving data...'
	symbols = [x.rstrip('\n') for x in open('symbols.txt','r').readlines()]
	for symb in symbols:
		save_to = '../data/data/' + symb + '.csv'
		try:
			url = construct_url('NASDAQ',symb,start,end)
			get_url(save_to,url)
		except:
			try:
				url = construct_url('NYSE',symb,start,end)
				get_url(save_to,url)
			except:
				print 'Cannot retrieve ' + symb + ' from Google Finance.'
	save_to = '../data/data/SP500.csv'
	url = 'https://www.google.com/finance/historical?q=NYSEARCA%3ASPY&startdate=' + start + '&enddate=' + end + '&output=csv'
	file = urllib2.urlopen(url)
	with open(save_to, 'wb') as output:
		output.write(file.read())
	print 'Successfully retrieved data.'	
	return

if __name__ == "__main__":
	main(sys.argv[1:])