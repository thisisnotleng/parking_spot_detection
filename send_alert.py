import requests


def alert_wrong_parking(spot_number):

    bot_token = 'bot_token'
    chat_id = 'chat_id'
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': f'There is a car park incorrectly at spot {spot_number}',
        'parse_mode': 'Markdown'  # Optional: 'Markdown' or 'HTML'
    }

    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(f"Response: {response.text}")
