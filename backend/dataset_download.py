import urllib.request
import zipfile
import os

IIIT_INDIC_URLS = {
    'Bengali': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Bengali.zip',
    'Gujarati': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Gujarati.zip',
    'Hindi': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Devanagari.zip',
    'Kannada': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Kannada.zip',
    'Malayalam': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Malayalam.zip',
    'Tamil': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Tamil.zip',
    'Telugu': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Telugu.zip',
    'Urdu': 'https://cvit.iiit.ac.in/images/Projects/IIIT_Indic_HW_Words/Urdu.zip'
}

def download_dataset(language, output_dir='../data/raw'):
    os.makedirs(output_dir, exist_ok=True)
    url = IIIT_INDIC_URLS[language]
    zip_path = os.path.join(output_dir, f'{language}.zip')
    
    print(f'Downloading {language} dataset...')
    urllib.request.urlretrieve(url, zip_path)
    
    print(f'Extracting {language} dataset...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, language))
    
    os.remove(zip_path)
    print(f'{language} dataset ready.')

if __name__ == '__main__':
    for lang in ['Hindi', 'Tamil', 'Telugu']:
        download_dataset(lang)
