"""Helper functions to interact with Firebase for storage and notifications."""

from pyfcm import FCMNotification
import firebase_admin
from firebase_admin import credentials, storage, firestore, messaging
import os
import datetime

# Initialize the Firebase Admin SDK
cred = credentials.Certificate(
    "intruder-detector-ef9ab-firebase-adminsdk-jqbsf-7694bf284d.json"
)
firebase_admin.initialize_app(
    cred, {"storageBucket": "intruder-detector-ef9ab.appspot.com"}
)
# Initialize Firestore
db = firestore.client()

fcm = FCMNotification(
    service_account_file="intruder-detector-ef9ab-firebase-adminsdk-jqbsf-7694bf284d.json",
    project_id="intruder-detector-ef9ab",
)


def upload_image(image_path, destination_blob_name):
    """Upload an image to Firebase Storage and return its public URL."""
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)
    blob.make_public()
    return blob.public_url


def save_url_to_firestore(url, date_time, document_path):
    """Persist ``url`` and ``date_time`` to Firestore at ``document_path``."""
    doc_ref = db.document(document_path)
    doc_ref.set({"image_url": url, "date_time": date_time})


def send_notification(user_id, doc_id):
    """Send a push notification to ``user_id`` about ``doc_id``."""
    registration_id_1 = "fMvhZMxpTZqmV5v-6xIEVo:APA91bGxaJn8DUeumYSrm9lE-832f8_KGqaWoE9GcAfNLYWiXkS8E3m58gdXtJxNVTzMhI3QNSIPInEBIRf2DZbZiKObRP2jFs6bMbCcdBMNIe8DqAfd6n1pvOMx3W7okqkuO2hYovpD"
    registration_id_2 = "d3mxt8FTQNCbkkybN58LfZ:APA91bEiTNOW0MAuE3zX8EaufLNeyB3GAWH4MQ6bFV2niPqrkdlAZcJo56aPYSc_XNPfSQiCTah9v61ml9t-jX5KsNkTtIKdOEmejecy2uK2bvaoV77Tmpr9I-z8o80UtneHEwReiU_o"
    message_title = "An intruder is detected"
    message_body = "Let see who is that"

    result = fcm.notify(
        fcm_token=registration_id_2,
        notification_title=message_title,
        notification_body=message_body,
        data_payload={"doc_id": doc_id, "user_id": user_id},
    )


if __name__ == "__main__":
    # Example usage:
    image_path = "uploads/faces_135_1_0.jpg"
    filename = os.path.basename(image_path)

    destination_blob_name = f"images/HodSqk9AoxhgG3SfgWVL/{filename}"
    document_path = (
        f"Users/HodSqk9AoxhgG3SfgWVL/intruders/{os.path.splitext(filename)[0]}"
    )

    # Upload the image and get the URL
    image_url = upload_image(image_path, destination_blob_name)

    # Get the current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print("Current date and time:", formatted_time)

    # Save the URL in Firestore
    save_url_to_firestore(image_url, formatted_time, document_path)

    send_notification("HodSqk9AoxhgG3SfgWVL", os.path.splitext(filename)[0])
