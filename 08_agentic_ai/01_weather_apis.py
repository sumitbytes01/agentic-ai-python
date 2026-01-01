from ollama import chat
import requests

def user_query():
    # Send the user's message to the ollama `chat` call
    instruction= input("Enter your query ➡️")

    response = chat(
        model='llama3.1', 
        messages=[{
            "role": 'user', 
            'content': instruction
            }])
    
    print("Here is the response => ", response.message.content)

def get_weather(city):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"the weather of the city {city} is {response.text}"
    
    return "Something must have gone wrong!!!"



user_query()
print(get_weather(input("enter the city name here:🤜 ")))