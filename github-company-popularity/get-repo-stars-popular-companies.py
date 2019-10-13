import requests
import csv

data = []
head = {'Authorization': 'token code'}

def saveData(body, company):
    for item in body["items"]:
        data.append([company, item["full_name"], item["stargazers_count"], item["watchers_count"], item["language"]])

def getNextUrl(headers):
    links = headers.get('link', None)
    if links is not None:
        individualLinks = links.split(",")
        firstLink = individualLinks[0].split(";")
        if "next" in firstLink[1]:
            nextPageUrl = firstLink[0][1:-1]
            return nextPageUrl
        else:
            return None
    return None

def getData(url):
    response = requests.get(url, headers=head)
    return response

def queryApi(url, company):
    content = getData(url)
    saveData(content.json(), company)
    nextUrl = getNextUrl(content.headers)
    if nextUrl:
        return queryApi(nextUrl, company)
    else:
        return None

company_lists = {
    "google" : ["google", "googlesamples"],
    "facebook" : ["facebook"],
    "apache" : ["apache"],
    "microsoft" : ["microsoft"],
    "mozilla" : ["mozilla"],
    "apple": ["apple"],
    "amazon": ["amzn", "amazonwebservices", "aws"]
}

for key, value in company_lists.items():
    for company in value:
        queryApi("https://api.github.com/search/repositories?q=org:{}&type=Repositories&per_page=100".format(company), key)

with open('../data/github-companies/companies2.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['Company', 'Repository', 'Stars', 'Watchers', 'Language'])
    for x in data:
        wr.writerow(x)