from selenium import webdriver
import csv

driver = webdriver.Chrome()

with open('data/magento2-modules.csv', 'r') as f:
    reader = csv.reader(f)
    module_list = list(reader)

for item in module_list:
    url = 'window.open("https://github.com/{0}")'.format(item[0])
    driver.execute_script(url)
