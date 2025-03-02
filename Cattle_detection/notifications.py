# notifications.py
import json
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from config import SERVICE_ACCOUNT_PATH, DEVICE_TOKEN

def send_fcm_alert(message: str, fcm_server_key: str, device_token: str) -> None:
    """Send a push notification via Firebase Cloud Messaging (FCM)."""
    headers = {
        "Authorization": f"key={fcm_server_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "to": device_token,
        "notification": {
            "title": "Barn Alert",
            "body": message,
            "sound": "default"
        }
    }

    try:
        response = requests.post(
            "https://fcm.googleapis.com/fcm/send",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        print("Alert sent successfully!")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")