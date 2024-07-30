import wikipediaapi
import pandas as pd

def get_wikipedia_text(titles):
    wiki_wiki = wikipediaapi.Wikipedia('MyApp/1.0 (https://myapp.example.com; myemail@example.com)')
    data = []

    for title in titles:
        page = wiki_wiki.page(title)
        if page.exists():
            data.append({'title': title, 'text': page.text})
        else:
            print(f"The page '{title}' does not exist.")

    return data

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename)

# titles = ['Opisthotropis hungtai','Megalocnidae',
#           'Black_baza', 'Qinling_panda', 'Humblot\'s heron',
#           "New Caledonian rail", "Arctocyon", "Reeves's pheasant",
#           "Sultan tit", "Yellow-throated marten"]
titles=[
"Game (2016 film)",
"Mother's Day (2016 film)",
"Indignation (film)",
"Fifty Shades of Black",
"Burn Your Maps",
"Anthropoid (film)",
"The Age of Shadows",
"Almost Friends (2016 film)",
"James & Alice",
"For the Love of Spock"
]
data = get_wikipedia_text(titles)
save_to_csv(data, 'test.xlsx')