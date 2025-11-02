from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import csv

url = "https://www.amazon.com/gp/browse.html?node=6563140011&ref_=nav_em_amazon_smart_home_0_2_8_2"

cService = webdriver.ChromeService(executable_path='C:\\Users\\Qasim\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe') # '/Users/bpfalz/Downloads/chromedriver' for my macbook
driver = webdriver.Chrome(service=cService)

driver.get(url)

productList=[]
productsdiv = driver.find_elements(By.XPATH, "//div[contains(@class, '_Y29ud_bxcGridColumn_J5gfU _Y29ud_bxcGridColumn1Of5_UoKNf')]")
for p in range(len(productsdiv) -1):
    pros = {}
    innerImg = productsdiv[p+1].find_element(By.TAG_NAME, "img")
    innera = productsdiv[p+1].find_element(By.TAG_NAME, "a")
    pros["img"] =innerImg.get_attribute('src') 
    pros["lines"] =innerImg.get_attribute('alt') 
    pros['url'] = innera.get_attribute('href')
    productList.append(pros)

filename = 'Amazon web scrabing/Amazon_smart.csv'
with open(filename, 'w', newline='') as f:
    w = csv.DictWriter(f,['url','img','lines','author'])
    w.writeheader()
    for pros in productList:
        w.writerow(pros)

driver.close()