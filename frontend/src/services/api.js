import axios from 'axios';

export const uploadImage = (formData) => {
  return axios.post('/api/post-upload-thyroid-cancer-image', formData);
};

export const getColumnFigureData = () => {
  return axios.get('/api/get-column-figure-data');
};