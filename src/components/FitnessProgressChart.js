import { Line } from 'react-chartjs-2';

const fitnessData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
  datasets: [
    {
      label: 'User Progress',
      data: [65, 75, 80, 90, 95],
      borderColor: 'rgba(75, 192, 192, 1)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      fill: true,
    },
  ],
};

const FitnessProgressChart = () => {
  return <Line data={fitnessData} />;
};

export default FitnessProgressChart;