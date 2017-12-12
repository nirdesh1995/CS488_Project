import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta


def compile_guardian(start_date = date(2000,1, 1),end_date=date(2000,2, 1)):

    ARTICLES_DIR = join('data' ,'articles')
    makedirs(ARTICLES_DIR, exist_ok=True)


    MY_API_KEY ='539d12aa-9089-4d6a-898b-65e3f26e56d5'
    API_ENDPOINT = 'http://content.guardianapis.com/search'
    parameters = {
        'from-date': "",
        'to-date': "",
        'order-by': "newest",
        'show-fields': 'all',
        'page-size': 200,
        'api-key': MY_API_KEY
    }




    all_days = range((end_date - start_date).days + 1)
    
    for daycount in all_days:
        dt = start_date + timedelta(days=daycount)
        datestr = dt.strftime('%Y-%m-%d')
        
        file_name = join(ARTICLES_DIR, datestr + '.json')
        
        if not exists(file_name):
            
            print("Dowloading articles for date: ", datestr)
            
            final = []
            parameters['from-date'] = datestr
            parameters['to-date'] = datestr
            current_page = 1
            total_pages = 1
            while current_page <= total_pages:
                #print("page==", current_page)
                parameters['page'] = current_page
                resp = requests.get(API_ENDPOINT, parameters)
                data = resp.json()
                final.extend(data['response']['results'])
                current_page += 1
                total_pages = data['response']['pages']

            with open(file_name, 'w') as f:
                #print("Writing to", file_name)
                f.write(json.dumps(final, indent=2))