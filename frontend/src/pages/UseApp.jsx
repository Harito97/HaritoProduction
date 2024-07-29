import React from 'react';
import Navbar from '../components/Navbar/Navbar';
import Footer from '../components/Footer/Footer';
import UseAppContent from '../contents/UseAppContent';

const UseApp = () => {
    return (
        <div>
            {/* Navbar */}
            <Navbar />

            {/* Main Content */}
            <UseAppContent />

            {/* Footer */}
            <Footer />
        </div>
    );
};

export default UseApp;