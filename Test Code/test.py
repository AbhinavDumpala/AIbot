

f=open('fear-ratings-0to1.dev.target.txt', encoding="utf8")

ls=f.readlines()

for line in ls:
	line=line.split('\t')
	#print(line[1])
	l1=line[1]
	l1=l1.split()

	st=''

	for word in l1:
		
		if '@' in word:
			continue
			st=st+word+" "
		else:
			st=st+word+" "

	print(st)



