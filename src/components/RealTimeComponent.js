import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

const socket = io("http://localhost:8000");

function RealTimeComponent() {
  const [prediction, setPrediction] = useState(null);
  
  useEffect(() => {
    socket.on('activity_response', (data) => {
      setPrediction(data.prediction);
    });
    
    return () => socket.disconnect();
  }, []);

  const sendActivity = () => {
    const activityData = [1, 0.5, 0.8];  // Replace with real data
    socket.emit('analyze_activity', { activity: activityData, feedback: "I feel great!" });
  };

  return (
    <div>
      <h2>Real-Time Activity Analysis</h2>
      <button onClick={sendActivity}>Send Activity Data</button>
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default RealTimeComponent;