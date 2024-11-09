import { Bar } from 'react-chartjs-2';

const goalData = {
  labels: ['Calories Burned', 'Workouts', 'Running Distance'],
  datasets: [
    {
      label: 'Your Progress',
      data: [80, 90, 100],
      backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)'],
    },
  ],
};

const GoalAchievementChart = () => {
  return <Bar data={goalData} />;
};

export default GoalAchievementChart;