import csv

data = ['First Item', 'Second Item', 'Third Item']
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in data:
        writer.writerow([i])