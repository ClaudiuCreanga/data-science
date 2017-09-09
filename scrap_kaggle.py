import scrapy
import json
import logging
from w3lib.url import add_or_replace_parameter
import re
class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.kaggle.com/kernels/all/0?sortBy=votes&after=false']
    page = 0

    def parse(self, response):
        data = json.loads(str(response.body, 'utf-8'))
        for item in data:
            finalData = {
                "language": item["languageName"],
                "comments": item["totalComments"],
                "votes": item["totalVotes"],
                "source": item["dataSources"][0]["name"],
                "medal": item["medal"],
                "date": item["scriptVersionDateCreated"]
            }
            yield finalData
        if data[len(data) - 1]["id"]:
            self.page += 20
            url = add_or_replace_parameter(response.url, 'after', data[len(data) - 1]["id"])
            url = re.sub(r"([0-9]){1,9}(?=\?)", str(self.page), url)
            yield scrapy.Request(url, self.parse)
