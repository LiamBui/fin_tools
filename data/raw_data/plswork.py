
f = open('20170408(1)/out.log','r')
g = open('20170408(1)/20170408(1).csv','w')

for row in f:
	if len(row) <= 91:
		continue
	row = row[49:]
	g.write(row)

f.close()
g.close()