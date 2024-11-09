import React from 'react';
import './App.css';
import FitnessProgressChart from './components/FitnessProgressChart';
import GoalAchievementChart from './components/GoalAchievementChart';
import RealTimeComponent from './components/RealTimeComponent';

function App() {
  return (
    <div className="App">
      <h1>Fitness App</h1>
      <FitnessProgressChart />
      <GoalAchievementChart />
      <RealTimeComponent />
    </div>
  );
}

export default App;