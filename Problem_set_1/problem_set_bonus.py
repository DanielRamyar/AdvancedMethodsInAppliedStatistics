import PyPDF2

pdfFileObj = open('authors-acknowledgements-v5.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

print(pdfReader.numPages)

pageObj = pdfReader.getPage(0)

text = pageObj.extractText()

to_loop = text.split()

temp_authors = []
for i in to_loop:
    if not i.isdigit():
        temp_authors.append(i)

del temp_authors[0:14]


authors = []
temp = ''
for i in temp_authors:
    if ',' in i:
        authors.append(temp)
        temp = ''
    else:
        temp = temp + i
    
### new page

pageObj = pdfReader.getPage(1)
text = pageObj.extractText()
to_loop = text.split()

temp_authors = []
for i in to_loop:
    if not i.isdigit():
        temp_authors.append(i)

temp = ''
for i in temp_authors:
    if ',' in i:
        authors.append(temp)
        temp = ''
    else:
        temp = temp + i

### new page 

pageObj = pdfReader.getPage(2)
text = pageObj.extractText()
to_loop = text.split()

temp_authors = []
for i in to_loop:
    if not i.isdigit():
        temp_authors.append(i)



del temp_authors[0:2]
del temp_authors[877:886]
del temp_authors[960:963]
del temp_authors[1031]
del temp_authors[874]
del temp_authors[954]
del temp_authors[1026]

temp = ''
for i in temp_authors:
    if ',' in i:
        authors.append(temp)
        temp = ''
    else:
        temp = temp + i

### new page 

pageObj = pdfReader.getPage(3)
text = pageObj.extractText()
to_loop = text.split()

temp_authors = []
for i in to_loop:
    if not i.isdigit():
        temp_authors.append(i)

del temp_authors[776:783]
del temp_authors[801:814]
del temp_authors[773]
del temp_authors[777]
del temp_authors[796]
del temp_authors[833:836]
del temp_authors[1178:1187]
del temp_authors[830]
del temp_authors[1174]
# print(temp_authors.index('AND'))
# print(temp_authors[1178:1187])

temp = ''
for i in temp_authors:
    if ',' in i:
        authors.append(temp)
        temp = ''
    else:
        temp = temp + i

### new page

pageObj = pdfReader.getPage(4)
text = pageObj.extractText()
to_loop = text.split()

temp_authors = []
for i in to_loop:
    if not i.isdigit():
        temp_authors.append(i)

del temp_authors[0:2]
del temp_authors[410:413]
del temp_authors[506:513]
del temp_authors[642:645]
del temp_authors[703:708]
del temp_authors[404]
del temp_authors[501]
del temp_authors[637]
del temp_authors[695]
del temp_authors[1171]
# print(temp_authors.index('AND'))
# print(temp_authors[703:708])

temp = ''
for i in temp_authors:
    if ',' in i:
        authors.append(temp)
        temp = ''
    else:
        temp = temp + i

print(authors)