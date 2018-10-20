# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:34:01 2018

@author: jaipe
"""
import matplotlib
import csv

FileDestination = 'googleplaystore.csv'

with open(FileDestination) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',') # creates a reader object
    line_count = 0
    rownumber = 0
    
    appname = []
    rating = []
    highrating = 0
    review = []
    installs = []
    price = []
    
    art = 0
    auto = 0
    beauty = 0
    
    #print
    for row in csv_reader: # this moves through each row in your csv file
        if rownumber == 102:
            break
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            rownumber +=1
        else:
            appname.append(row[0])
            
            rating.append(row[2])
            review.append(int(row[3]))
            
            row[5] = row[5].replace(",", "")
            installs.append(int(row[5][:-1]))
            
            row[7] = row[7].replace("$", "")
            price.append(float(row[7]))
            
            if float(row[2]) > 4:
                highrating +=1
            
            if row[9] == 'Art & Design':
                art += 1
            if row[9] == 'Auto & Vehicles':
                auto += 1
            if row[9] == 'Beauty':
                beauty += 1
            
            line_count += 1
            rownumber +=1
    print(f'Processed {line_count} lines.')
    
    #for x in appname:
     #   print(x)
    
    #for y in rating:
     #   print(y)
     
#print(installs[4])
bin1 = [0,1,2,3,4,5]  
print(highrating)

plt = matplotlib.pyplot
plt.hist(rating,bin1, histtype='bar', rwidth =0.8)
plt.xlablel =('ratings')
plt.ylablel =('counts')
plt.title('Histogram of game ratings')
plt.legend()
plt.show()  
    
plt2 = matplotlib.pyplot
plt2.scatter(rating, installs,color = 'k')
plt2.xlablel =('ratings')
plt2.ylablel =('installs')
plt2.title('Installs versus ratings')
plt2.legend()
plt2.show()  

plt3 = matplotlib.pyplot
plt3.scatter(rating, review,color = 'k')
plt3.xlablel =('ratings')
plt3.ylablel =('reviews')
plt3.title('Reviews versus ratings')
plt3.legend()
plt3.show()

plt4 = matplotlib.pyplot
plt4.scatter(review, installs,color = 'k')
plt4.xlablel =('reviews')
plt4.ylablel =('installs')
plt4.title('Installs versus Reviews')
plt4.legend()
plt4.show()  
    
#print(art)
#print(auto)
#print(beauty)

categories = [ 'art' ,'auto', 'beauty']
category_values = [ art, auto, beauty]
cols = ['b','r','g'] 

plt5 = matplotlib.pyplot
plt5.pie(category_values, labels = categories, colors =cols)
    