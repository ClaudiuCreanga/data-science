import scrapy
import json
import logging
from w3lib.url import add_or_replace_parameter
import re
from scrapy.selector import HtmlXPathSelector

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.kaggle.com/rankings.json?group=competitions&page=1&pageSize=20']
    page = 0
    pageSize = 20
    profiles = []

    def parse(self, response):
        data = json.loads(str(response.body, 'utf-8'))
        for item in data["list"]:
            id = item["userUrl"]
            if id not in self.profiles:
                self.profiles.append(id)
                user_url = "https://www.kaggle.com" + id
                yield scrapy.Request(user_url, self.parseLocation)

        self.page += 1
        url = add_or_replace_parameter(response.url, 'page', self.page)
        url = add_or_replace_parameter(url, 'pageSize', self.pageSize)
        yield scrapy.Request(url, self.parse)

    def parseLocation(self, response):
        hxs = HtmlXPathSelector(response)
        location = hxs.select('//div[@class="site-layout__main-content"]').extract()
        location = "".join(location[0].splitlines())
        location_city = re.search('(?<=city":)(.*?),"gitHubUserName', location).group(1)
        location_country = re.search('(?<=country":)(.*?),"region', location).group(1)
        twitter = re.search('(?<=twitterUserName":)(.*?),"linkedInUrl', location).group(1)
        join = re.search('(?<=userJoinDate":)(.*?),"performanceTier', location).group(1)
        yield dict({"city" : location_city, "country": location_country, "twitter": twitter, "join_date": join})