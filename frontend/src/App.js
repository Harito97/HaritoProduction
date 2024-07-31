import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import UseApp from './pages/UseApp';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/use_app" element={<UseApp />} />
        <Route path="/" element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;