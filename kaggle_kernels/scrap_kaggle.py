import scrapy
import json
import logging
from w3lib.url import add_or_replace_parameter
import re
class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.kaggle.com/kernels/all/20?sortBy=date&after=false']
    page = 0
    ids = []

    def parse(self, response):
        data = json.loads(str(response.body, 'utf-8'))
        for item in data:
            finalData = {
                "language": item["languageName"],
                "comments": item["totalComments"],
                "votes": item["totalVotes"],
                "medal": item["medal"],
                "id": item["id"],
                "date": item["scriptVersionDateCreated"]
            }
            id = item["id"]
            yield finalData

        if id not in self.ids:
            self.ids.append(id)
        else:
            logging.info("The id is duplicate, stop here")
            return

        if data[len(data) - 1]["id"]:
            self.page += 20
            if self.page > 1000:
                self.page = 1000
            url = add_or_replace_parameter(response.url, 'after', data[len(data) - 1]["id"])
            url = re.sub(r"([0-9]){1,9}(?=\?)", str(self.page), url)
            yield scrapy.Request(url, self.parse)
