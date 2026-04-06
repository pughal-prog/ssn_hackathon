"""
Module 6: SMS Alert System via Twilio
Free trial at twilio.com — gives you $15 credit
"""
from twilio.rest import Client

# Get these from twilio.com/console after signing up (free)
ACCOUNT_SID  = "YOUR_TWILIO_ACCOUNT_SID"
AUTH_TOKEN   = "YOUR_TWILIO_AUTH_TOKEN"
FROM_NUMBER  = "YOUR_TWILIO_PHONE_NUMBER"   # e.g. +15551234567

def send_sms_alert(to_number: str, lake_name: str, risk: str,
                   explanation: str, eta: str = None):
    """
    Send SMS alert for a HIGH risk lake.
    to_number: recipient phone e.g. "+919876543210"
    """
    if risk != "HIGH":
        return {"status": "skipped", "reason": "Risk is not HIGH"}

    eta_line = f"\n⏱️ Critical in: {eta}" if eta else ""
    message  = (
        f"🚨 GLOF EARLY WARNING ALERT 🚨\n"
        f"Lake: {lake_name}\n"
        f"Risk Level: {risk}\n"
        f"Reason: {explanation}"
        f"{eta_line}\n"
        f"Take immediate precautionary action.\n"
        f"— GLOF Early Warning System"
    )

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        msg = client.messages.create(body=message, from_=FROM_NUMBER, to=to_number)
        return {"status": "sent", "sid": msg.sid}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def send_bulk_alerts(scored_df, to_number: str):
    """Send alerts for all HIGH risk lakes in the dataset."""
    high_risk = scored_df[scored_df['risk'] == 'HIGH']
    results = []
    for _, row in high_risk.iterrows():
        result = send_sms_alert(
            to_number  = to_number,
            lake_name  = row['name'],
            risk       = row['risk'],
            explanation= row.get('explanation', ''),
        )
        results.append({"lake": row['name'], **result})
    return results
