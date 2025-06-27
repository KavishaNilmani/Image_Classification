import requests

def get_description(category):
    if category not in ['mammal', 'bird', 'fish', 'plant']:
        return "🔍 Data not available for this category."

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{category}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data.get('extract', '📄 Description not found.')
    except:
        return "⚠️ Error fetching data from Wikipedia."
