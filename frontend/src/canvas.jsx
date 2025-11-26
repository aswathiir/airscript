import React, { useRef, useEffect, useState } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import axios from 'axios';

const Canvas = ({ userId }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [strokes, setStrokes] = useState([]);
  const [currentStroke, setCurrentStroke] = useState([]);
  const [recognizedText, setRecognizedText] = useState('');
  const [isDrawing, setIsDrawing] = useState(false);
  
  const BACKEND_URL = 'http://localhost:8000';

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    hands.onResults(onResults);

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 640,
      height: 480
    });
    camera.start();

    return () => {
      camera.stop();
    };
  }, []);

  const onResults = (results) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const indexFingerTip = landmarks[8];
      
      const x = indexFingerTip.x * canvas.width;
      const y = indexFingerTip.y * canvas.height;

      ctx.fillStyle = '#00FF00';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();

      const thumbTip = landmarks[4];
      const indexTipY = landmarks[8].y;
      const thumbTipY = thumbTip.y;
      
      if (Math.abs(indexTipY - thumbTipY) < 0.05) {
        if (isDrawing) {
          setIsDrawing(false);
          setStrokes(prev => [...prev, currentStroke]);
          setCurrentStroke([]);
        }
      } else {
        setIsDrawing(true);
        const point = {
          x: x,
          y: y,
          timestamp: Date.now()
        };
        setCurrentStroke(prev => [...prev, point]);
      }
    }

    drawAllStrokes(ctx);
  };

  const drawAllStrokes = (ctx) => {
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    [...strokes, currentStroke].forEach(stroke => {
      if (stroke.length < 2) return;
      
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i].x, stroke[i].y);
      }
      ctx.stroke();
    });
  };

  const handleRecognize = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/recognize`, {
        user_id: userId,
        strokes: strokes
      });
      
      setRecognizedText(response.data.recognized_text);
    } catch (error) {
      console.error('Recognition error:', error);
      alert('Recognition failed. Check backend connection.');
    }
  };

  const handleClear = () => {
    setStrokes([]);
    setCurrentStroke([]);
    setRecognizedText('');
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
      <video ref={videoRef} style={{ display: 'none' }} />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ border: '2px solid #333', borderRadius: '8px' }}
      />
      
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={handleRecognize}
          style={{
            padding: '12px 24px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Recognize
        </button>
        
        <button
          onClick={handleClear}
          style={{
            padding: '12px 24px',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Clear
        </button>
      </div>

      {recognizedText && (
        <div style={{
          padding: '16px',
          backgroundColor: '#f0f0f0',
          borderRadius: '8px',
          fontSize: '20px',
          fontWeight: 'bold'
        }}>
          Recognized: {recognizedText}
        </div>
      )}
    </div>
  );
};

export default Canvas;
