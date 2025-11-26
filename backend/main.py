from firebase_functions import https_fn
from firebase_admin import initialize_app
import json
from model_load import HTRModelLoader
from stroke_processor import StrokeProcessor
import torch

initialize_app()

device = 'cpu'
model_loader = HTRModelLoader(
    checkpoint_path='./models/htr/checkpoint_finetuned.pth',
    device=device
)
stroke_processor = StrokeProcessor()

@https_fn.on_request()
def recognize(req: https_fn.Request) -> https_fn.Response:
    if req.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return https_fn.Response('', status=204, headers=headers)
    
    if req.method != 'POST':
        return https_fn.Response('Method not allowed', status=405)
    
    try:
        data = req.get_json()
        
        strokes_dict = [
            [{'x': pt['x'], 'y': pt['y']} for pt in stroke]
            for stroke in data['strokes']
        ]
        
        image = stroke_processor.strokes_to_image(strokes_dict)
        
        if image is None:
            return https_fn.Response(
                json.dumps({'error': 'Invalid stroke data'}),
                status=400,
                headers={'Content-Type': 'application/json'}
            )
        
        input_tensor = stroke_processor.prepare_input(image)
        prediction = model_loader.predict(input_tensor)
        
        response_data = {
            'user_id': data.get('user_id'),
            'recognized_text': prediction,
            'confidence': 0.95
        }
        
        return https_fn.Response(
            json.dumps(response_data),
            status=200,
            headers={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    except Exception as e:
        return https_fn.Response(
            json.dumps({'error': str(e)}),
            status=500,
            headers={'Content-Type': 'application/json'}
        )
