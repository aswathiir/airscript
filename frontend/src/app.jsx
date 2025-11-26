import React, { useState } from 'react';
import Canvas from './Canvas';

function App() {
  const [userId] = useState(`user${Date.now()}`);

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#1a1a1a',
      color: 'white',
      padding: '40px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '48px', margin: '0 0 10px 0' }}>
          Air Canvas
        </h1>
        <p style={{ fontSize: '18px', color: '#aaa' }}>
          Gesture-based handwriting recognition
        </p>
        <p style={{ fontSize: '14px', color: '#666' }}>
          User ID: {userId}
        </p>
      </header>

      <main style={{ display: 'flex', justifyContent: 'center' }}>
        <Canvas userId={userId} />
      </main>

      <footer style={{
        marginTop: '60px',
        textAlign: 'center',
        fontSize: '14px',
        color: '#666'
      }}>
        <p>Pinch thumb and index finger to stop drawing</p>
        <p>Draw in the air with your index finger</p>
      </footer>
    </div>
  );
}

export default App;
