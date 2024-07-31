import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { getColumnFigureData } from '../../services/api';

const ColumnFigure = () => {
  const [labels, setLabels] = useState([]);
  const [probabilities, setProbabilities] = useState([]);

  useEffect(() => {
    getColumnFigureData()
      .then((response) => {
        setLabels(response.data.labels);
        setProbabilities(response.data.probabilities);
      })
      .catch((error) => {
        console.error('Error fetching column figure data:', error);
      });
  }, []);

  const data = {
    labels: labels,
    datasets: [
      {
        label: 'Probabilities',
        data: probabilities,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  return <Bar data={data} options={options} />;
};

export default ColumnFigure;
