# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:16:30 2019

@author: Ahsan
"""

#Python code to illustrate parsing of XML files 
# importing the required modules 
import csv 
import os
import requests 
import xml.etree.ElementTree as ET 
  
def loadRSS(): 
  
    # url of rss feed 
    url = 'http://www.hindustantimes.com/rss/topnews/rssfeed.xml'
  
    # creating HTTP response object from given url 
    resp = requests.get(url) 
  
    # saving the xml file 
    with open('topnewsfeed.xml', 'wb') as f: 
        f.write(resp.content) 
          
  
def parseXML(xmlfile): 
  
    # create element tree object 
    tree = ET.parse(xmlfile) 
  
    # get root element 
    root = tree.getroot() 
  
    # create empty list for news items 
    item = {} 
  
    
    item['filename'] = root.find('./filename').text
    item['width'] = root.find('./size/width').text
    item['height'] = root.find('./size/height').text
    item['depth'] = root.find('./size/depth').text
    item['name'] = root.find('./object/name').text
    item['pose'] = root.find('./object/pose').text
    item['truncated'] = root.find('./object/truncated').text
    item['occluded'] = root.find('./object/occluded').text
    
    item['xmin'] = root.find('./object/bndbox/xmin').text
    item['ymin'] = root.find('./object/bndbox/ymin').text
    item['xmax'] = root.find('./object/bndbox/xmax').text
    item['ymax'] = root.find('./object/bndbox/ymax').text
    
    
    item['difficult'] = root.find('./object/difficult').text
    
    #print( item )
    
    """
    # iterate news items 
    for item in root.findall('./channel/item'): 
  
        # empty news dictionary 
        news = {} 
  
        # iterate child elements of item 
        for child in item: 
  
            # special checking for namespace object content:media 
            if child.tag == '{http://search.yahoo.com/mrss/}content': 
                news['media'] = child.attrib['url'] 
            else: 
                news[child.tag] = child.text.encode('utf8') 
  
        # append news dictionary to news items list 
        newsitems.append(news) 
      
    # return news items list 
    """
    return item 
  
  
def savetoCSV(newsitems, filename): 
  
    # specifying the fields for csv file 
    fields = ['filename', 'width', 'height', 'depth', 'name', 'pose', 'truncated', 'occluded', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult'] 
  
    # writing to csv file 
    with open(filename, 'w', newline='') as csvfile: 
  
        # creating a csv dict writer object 
        writer = csv.DictWriter(csvfile, fieldnames = fields) 
  
        # writing headers (field names) 
        writer.writeheader() 
  
        # writing data rows 
        writer.writerows(newsitems) 
  
      
def main(): 
    # load rss from web to update existing xml file 
    #loadRSS() 
  
    root_dir = 'data/annotations/xmls/'
    
    items = []
    for filename in os.listdir(root_dir):
        item = parseXML(root_dir + filename)
        items.append(item)
        
        
    
    #print(items)
        
    # parse xml file 
    #newsitems = parseXML('topnewsfeed.xml') 
  
    # store news items in a csv file 
    savetoCSV(items, 'annotations.csv') 
      
      
if __name__ == "__main__": 
  
    # calling main function 
    main()