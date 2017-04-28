from github import Github
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


#g = Github("ClaudiuCreanga", "a840a89635d7caf657847fc207205f41b688a238")

magento2_data = pd.read_csv('data/m2-data-dates.csv', sep=',',header=None, names = ["Repo", "Start Date2", "Last Commit"])
magento1_data = pd.read_csv('data/m1-data-dates.csv', sep=',',header=None, names = ["Repo", "Start Date", "Last Commit"])
# horizontalStack = pd.concat([magento1_data[["Start Date"]], magento2_data[["Start Date"]]], axis=1)
magento2_start_dates = magento2_data["Start Date2"].apply(lambda row: datetime.datetime.fromtimestamp(int(row)).strftime('%Y-%m')).value_counts()
magento1_start_dates = magento1_data["Start Date"].apply(lambda row: datetime.datetime.fromtimestamp(int(row)).strftime('%Y-%m')).value_counts()
magento2_start_dates_sorted =magento2_start_dates.sort_index()
magento1_start_dates_sorted =magento1_start_dates.sort_index()
# magento1_start_dates_sorted = magento1_start_dates_sorted.to_frame()
# magento2_start_dates_sorted = magento2_start_dates_sorted.to_frame()
# magento2_start_dates_sorted.index.name = 'dateq'
print(magento1_start_dates_sorted)
magento2_start_dates_sorted.to_csv("out1.csv", sep='\t', encoding='utf-8')
magento1_start_dates_sorted.to_csv("out2.csv", sep='\t', encoding='utf-8')
result = pd.DataFrame(magento1_start_dates_sorted).reset_index().merge(pd.DataFrame(magento2_start_dates_sorted.reset_index()), left_on="Start Date", right_on="Start Date2", how='outer').set_index('index')
result.to_csv("out.csv", sep='\t', encoding='utf-8')

magento2_start_dates_sorted.plot()
magento1_start_dates_sorted.plot()




# da = g.search_code("\Magento\Framework\Component\ComponentRegistrar in:file language:xml")
# print(da)
# # for item in da:
# #     print(item)
# pulls_count = 0
# # Fix no count available on pulls list
# for _ in da:
#     pulls_count += 1
#
# print(pulls_count)