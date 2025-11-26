import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate('../deployment/service-account-key.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'aircanvas-prod.appspot.com'
})

bucket = storage.bucket()

blob = bucket.blob('models/htr/checkpoint_finetuned.pth')
blob.upload_from_filename('../models/htr/checkpoint_finetuned.pth')

print('Model uploaded to Firebase Storage')
