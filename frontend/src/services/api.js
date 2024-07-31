import axios from 'axios';

export const uploadImage = (formData) => {
  // return axios.post('/api/post-upload-thyroid-cancer-image', formData);
  return axios.post('http://localhost:8000/api/post-upload-thyroid-cancer-image', formData);
};

export const getColumnFigureData = () => {
  return axios.get('/api/get-column-figure-data');
};

